# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates() 
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # the purpose of this function is to run self.iterations rounds 
        # of value iteration, returning nothing.
        # V_{k+1} = max_a sum_{s'} T(s,a,s') [ R(s,a,s') + \gamma V_k(s')]

        
        for i in range(self.iterations):
            v_k = self.values.copy()
            for state in self.mdp.getStates():
                self.values[state] = self.v_k_plus_1(state, v_k)

            
    def v_k_plus_1(self, state, v_k):
        best = -float('inf')

        for action in self.mdp.getPossibleActions(state):
            curr_v = 0
            for (s_prime, t) in self.mdp.getTransitionStatesAndProbs(state, action):
                r = self.mdp.getReward(state, action, s_prime)
                curr_v += t * (r + self.discount * v_k[s_prime])
            if curr_v > best:
                best = curr_v

        if best == -float('inf'):
            return 0

        return best


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # Q*(s,a) = sum_{s'}T(s,a,s') [ R(s,a,s') + \gamma V*(s') ]

        gamma = self.discount
        q_s_a = 0

        for (next_state, t) in self.mdp.getTransitionStatesAndProbs(state, action):
            r = self.mdp.getReward(state, action, next_state)
            q_s_a += t*(r + gamma * self.getValue(next_state))

        return q_s_a


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        best_action = None
        best_action_val = -float('inf')

        for action in self.mdp.getPossibleActions(state):
            if self.computeQValueFromValues(state, action) > best_action_val:
                best_action = action
                best_action_val = self.computeQValueFromValues(state, action)


        return best_action
            

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            state = self.mdp.getStates()[i % len(self.mdp.getStates())]
            self.values[state] = self.update_v(state)

            
    def update_v(self, state):
        best = -float('inf')

        for action in self.mdp.getPossibleActions(state):
            curr_v = 0
            for (s_prime, t) in self.mdp.getTransitionStatesAndProbs(state, action):
                r = self.mdp.getReward(state, action, s_prime)
                curr_v += t * (r + self.discount * self.values[s_prime])
            if curr_v > best:
                best = curr_v

        if best == -float('inf'):
            return 0

        return best



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def youMayUpdateONCE(self, state):
        best = -float('inf')

        for action in self.mdp.getPossibleActions(state):
            curr_v = 0
            for (s_prime, t) in self.mdp.getTransitionStatesAndProbs(state, action):
                r = self.mdp.getReward(state, action, s_prime)
                curr_v += t * (r + self.discount * self.values[s_prime])
            if curr_v > best:
                best = curr_v

        if best == -float('inf'):
            return 0

        return best


    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states.
            # When you compute predecessors of a state, make sure to store them in a set, not a list, to avoid duplicates.
        states = self.mdp.getStates()
        predecessors = {}

        for state in states:
            for action in self.mdp.getPossibleActions(state):
                for (s_prime, t) in self.mdp.getTransitionStatesAndProbs(state, action):
                    if s_prime not in predecessors.keys():
                        predecessors[s_prime] = set()
                    predecessors[s_prime].add(state)

        # Initialize an empty priority queue.
            # Please use util.PriorityQueue in your implementation. The update method in this class will likely be useful; 
            # look at its documentation.
        pq = util.PriorityQueue()

        # For each non-terminal state s, do: (note: to make the autograder work for this question, you must iterate over 
        # states in the order returned by self.mdp.getStates())
        for s in states:
            if not self.mdp.isTerminal(s):
                # Find the absolute value of the difference between the current value of s in self.values and the highest Q-value 
                # across all possible actions from s (this represents what the value should be); call this number diff. Do NOT 
                # update self.values[s] in this step.
                best_q = -float('inf')
                for action in self.mdp.getPossibleActions(s):
                    q = self.computeQValueFromValues(s, action)
                    best_q = q if q > best_q else best_q
                diff = abs(self.getValue(s) - best_q)

                # Push s into the priority queue with priority -diff (note that this is negative). We use a negative because the 
                # priority queue is a min heap, but we want to prioritize updating states that have a higher error.
                pq.push(s, priority = -diff)


        # For iteration in 0, 1, 2, ..., self.iterations - 1, do: 
        for i in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if pq.isEmpty():
                break
            # Pop a state s off the priority queue.
            s = pq.pop()
            # Update the value of s (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(s):
                self.values[s] = self.youMayUpdateONCE(s)
            # For each predecessor p of s, do: 
            for p in predecessors[s]:
                # Find the absolute value of the difference between the current value of p in self.values and the highest 
                # Q-value across all possible actions from p (this represents what the value should be); call this number 
                # diff. Do NOT update self.values[p] in this step.
                best_q = -float('inf')
                for action in self.mdp.getPossibleActions(p):
                    q = self.computeQValueFromValues(p, action)
                    best_q = q if q > best_q else best_q
                diff = abs(self.getValue(p) - best_q)
                # If diff > theta, push p into the priority queue with priority -diff (note that this is negative), as long as 
                # it does not already exist in the priority queue with equal or lower priority. As before, we use a negative 
                # because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
                if diff > self.theta:
                    pq.update(p, -diff)