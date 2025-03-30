import logging
from typing import List
import itertools
import math

from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Value import Value
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import LinearAdditiveUtilitySpace
from tudelft_utilities_logging.Reporter import Reporter

from agents.group60_agent.utils.opponent_model import OpponentModel


class FinalPhasePlan:

    def __init__(self, profile, opponent_model, bids_left, logger):
        self.logger: Reporter = logger
        self.opponent_model: OpponentModel = opponent_model
        self.profile: LinearAdditiveUtilitySpace = profile
        self.all_bids: List[BidInfo] = self.sensible_bids_ordered()
        self.EU: float = 0.0
        self.plan: List[BidInfo] = []
        self.bids_executed: int = 0

        self.gca(bids_left)

    def gca(self, deadline):
        """
        Returns a sequence of bids which maximizes EU (expected utility)
        """
        self.logger.log(logging.INFO, "Starting GCA algorithm...")
        self.all_bids = [bid for bid in self.all_bids if not bid.selected]
        EU = float(self.profile.getUtility(self.profile.getReservationBid()) if self.profile.getReservationBid() is not None else 0)

        for i in range(deadline):
            best_EU = EU
            best_EU_bid = None
            EU_suf = EU
            p_pref = 1.0

            # find optimal bid to add to the plan
            for bid in self.all_bids:

                if bid.selected:
                    EU_suf = EU_suf - p_pref * bid.p * bid.u
                    p_pref = p_pref * (1.0 - bid.p)
                else:
                    EU_cur = EU + p_pref * bid.p * bid.u - bid.p * EU_suf
                    if EU_cur > best_EU:
                        best_EU = EU_cur
                        best_EU_bid = bid

            # check if it makes sense to make more bids
            if best_EU == EU:
                break

            # add best option to the plan
            best_EU_bid.selected = True
            EU = best_EU

        self.plan = [bid for bid in self.all_bids if bid.selected]
        self.EU = EU
        self.bids_executed = 0

        self.logger.log(logging.INFO, f"Optimal plan: {self.plan}")

    def next_bid(self):
        """
        Executes next bid in the plan
        """
        next_bid = self.plan[self.bids_executed]
        self.bids_executed = self.bids_executed + 1
        self.EU = (self.EU - next_bid.payoff()) / (1.0 - next_bid.p)
        if self.bids_executed == len(self.plan):
            self.gca(1)
        return next_bid.bid

    def current_EU(self):
        """
        Returns EU of the remaining part of the plan
        """
        self.logger.log(logging.INFO, f"EU: {self.EU}, bids in plan left: {len(self.plan) - self.bids_executed}")
        return self.EU

    def is_sensible(self, issue: str, value: Value) -> bool:
        """
        Returns False if value is not sensible for us nor for the opponent, and True otherwise
        """
        return (self.profile._issueUtilities[issue].getUtility(value) > 0.4 or
                self.opponent_model.issue_estimators[issue].get_value_utility(value) > 0.4)

    def sensible_bids_ordered(self):
        """
        Returns list of all sensible bids ordered by decreasing utility
        """
        self.logger.log(logging.INFO, "Getting useful bids...")

        issue_values = self.profile.getDomain().getIssuesValues()
        issues = issue_values.keys()

        useful_issue_values: dict[str, List[Value]] = {}
        for issue, values in issue_values.items():
            useful_values = [value for value in values if self.is_sensible(issue, value)]
            useful_issue_values[issue] = useful_values

        value_combinations = itertools.product(*list(useful_issue_values.values()))

        bids = [Bid(dict(zip(issues, combination))) for combination in value_combinations]

        bids = [BidInfo(bid, self.profile.getUtility(bid), self.opponent_model.get_predicted_utility(bid)) for bid in bids]

        bids.sort(reverse=True)

        self.logger.log(logging.INFO, f"Found {len(bids)} useful bids out of total {math.prod(lst.size() for lst in issue_values.values())}")

        return bids


class BidInfo:
    """
    Encapsulates useful information about a bid
    """
    bid: Bid = None
    u: float = 0
    p: float = 0
    selected: bool = False

    def __init__(self, bid: Bid, u, p):
        self.bid = bid
        self.u = float(u)
        self.p = float(p)
        self.selected = False

    def __lt__(self, other):
        return self.u < other.u

    def payoff(self):
        return self.u * self.p

    def __repr__(self):
        return f"{self.__class__.__name__}(bid={self.bid}, u={self.u}, p={self.p})"