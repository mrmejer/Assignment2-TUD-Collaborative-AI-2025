import logging
from random import randint
from time import time
from typing import cast
from random import sample
from decimal import Decimal

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel


class TemplateAgent(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

        self.initial_phase = 0
        self.discussion_phase = 1
        self.concession_phase = 2
        self.phase_boundaries = [0.15, 0.4, 0.9]
        self.reservation_value = None

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            reservation_bid = self.profile.getReservationBid()
            self.reservation_value = self.profile.getUtility(reservation_bid) if reservation_bid is not None else 0
            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn

            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Template agent for the ANL 2022 competition"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """
        Here, we divide the negotiation in multiple phases. In the initial phase, the idea is to learn more
        about our opponent's preferences. Therefore, we always bid the highest utility for us, and only
        accept an offer from the opponent if the overall utility of that bid for us is above 0.9.
        In the discussion phase, the idea is to get the best utility possible. We randomly select a uniform
        outcome from a time dependent-set to make bids. We only accept an offer if it has an utility of above 0.8.
        In the concession phase, the idea is to reach an agreement while still trying to get a good value for
        the utility. For this, we used the consideration phase idea described in Akiyuki Mori Takayuki Ito. 2016. Atlas3: Anegotiating Agent Based on Expecting
        Lower Limit of Concession Function. 169–173.
        """
        # Deciding whether to accept
        t = self.progress.get(time() * 1000)
        progress = t
        action = None

        if progress < self.phase_boundaries[self.initial_phase]:
            if self.accept_const(0.9, self.last_received_bid):
                action = Accept(self.me, self.last_received_bid)
            else:
                action = Offer(self.me,  self.best_bid())

        elif progress < self.phase_boundaries[self.discussion_phase]:
            if self.accept_const(0.8, self.last_received_bid):
                action = Accept(self.me, self.last_received_bid)
            else:
                action = Offer(self.me,  self.sample_bid_above_time_bound(t))

        elif progress < self.phase_boundaries[self.concession_phase]:
            if self.accept_Atlas3(self.last_received_bid, progress):
                action = Accept(self.me, self.last_received_bid)
            else:
                action = Offer(self.me, self.random_bid_above_best_join_utility(t))

        self.send_action(action)

    def accept_const(self, alpha: float, bid: Bid) -> bool:
        """Accept if the utility offered by the opponent is higher than a constant alpha"""
        if bid is None:
            return False
        return self.profile.getUtility(bid) > alpha

    def accept_Atlas3(self, bid: Bid, t: float) -> bool:
        """
        Accepting strategy implemented from Akiyuki Mori Takayuki Ito. 2016. Atlas3: Anegotiating Agent Based on Expecting
        Lower Limit of Concession Function. 169–173.
        """
        f_omega_bestOffered = Decimal(max([self.profile.getUtility(opponentBid) for opponentBid in self.opponent_model.offers]))
        f_omega_reserve = self.reservation_value

        u_CH = max(f_omega_reserve, f_omega_bestOffered) # if we're conceder and they are hardliner
        u_HH = f_omega_reserve # if both are hardliner we won't ever get anything better than reserve at the end
        u_HC = Decimal(1) # if we're hardliner and they're conceder we assume we can get max utility
        u_CC = Decimal('0.5') * u_CH + Decimal('0.5') * u_HC # if we're conceders we assume each can concede with equal probability

        q = 1 / (1 + ((u_CH - u_HH) / (u_HC - u_CC))) # q \in [0, 1]
        E_u_final = q * u_CH + (1 - q) * u_CC
        alpha = 1 - Decimal(t) * (1 - Decimal(E_u_final))

        return self.profile.getUtility(bid) > alpha

    def random_bid_above_best_join_utility(self, t: float):
        """Computes a time dependent joint utility distribution,
        fins the best possible joint utility from it (corresponding to an offer favourable to both parties) - f_omega_maxJoint
        and then randomly samples a bid from among bids that have larger utility that f_omega_maxJoint
        """
        joint_util = lambda omega: ((Decimal('1.8') - Decimal('0.3') * (Decimal(t)**2)) * self.profile.getUtility(omega)
                                   + Decimal(self.opponent_model.get_predicted_utility(omega)))
        domain = self.profile.getDomain()
        all_bids = AllBidsList(domain)

        u_joint = [joint_util(all_bids.get(i)) for i in range(0, all_bids.size())]
        f_omega_maxJoint = max(u_joint)

        return sample([all_bids.get(i) for i in range(0, all_bids.size()) if self.profile.getUtility(all_bids.get(i)) >= f_omega_maxJoint], k = 1)[0]

    def best_bid(self):
        """
        finding omega_best - greedily offer with the highest utility for us, regardless of opponent's preferences
        """
        domain = self.profile.getDomain()
        all_bids = AllBidsList(domain)
        best_bid_score = 0
        best_bid = None
        # take 500 attempts to find a bid according to a heuristic score
        for _ in range(500):
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            bid_score = self.profile.getUtility(bid)
            if bid_score > best_bid_score:
                best_bid_score, best_bid = bid_score, bid
        return best_bid

    def sample_bid_above_time_bound(self, t):
        """
        This method is called to generate random bids in the discussion phase of the negotiation.
        The bids have a lower bound that it's time-dependent. The lower bound decreases for each bid, up
        until a set lower bound of the maximum between our reservation bid and a utility of 0.6.
        """
        domain = self.profile.getDomain()
        all_bids = AllBidsList(domain)
        omegas = [all_bids.get(i) for i in range(0, all_bids.size() - 1)]
        omegas_filtered = [omega for omega in omegas if self.profile.getUtility(omega) >
                           max((self.phase_boundaries[self.discussion_phase] - self.phase_boundaries[self.initial_phase] * t**2),
                               max(self.reservation_value, 0.6))]

        return sample(omegas_filtered, k=1)[0]  # uniformly sample

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################
    def find_bid(self) -> Bid:
        # compose a list of all possible bids
        domain = self.profile.getDomain()
        all_bids = AllBidsList(domain)

        best_bid_score = 0.0
        best_bid = None

        # take 500 attempts to find a bid according to a heuristic score
        for _ in range(500):
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            bid_score = self.score_bid(bid)
            if bid_score > best_bid_score:
                best_bid_score, best_bid = bid_score, bid

        return best_bid

    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
        """Calculate heuristic score for a bid

        Args:
            bid (Bid): Bid to score
            alpha (float, optional): Trade-off factor between self interested and
                altruistic behaviour. Defaults to 0.95.
            eps (float, optional): Time pressure factor, balances between conceding
                and Boulware behaviour over time. Defaults to 0.1.

        Returns:
            float: score
        """
        progress = self.progress.get(time() * 1000)

        our_utility = float(self.profile.getUtility(bid))

        time_pressure = 1.0 - progress ** (1 / eps)
        score = alpha * time_pressure * our_utility

        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
            score += opponent_score

        return score
