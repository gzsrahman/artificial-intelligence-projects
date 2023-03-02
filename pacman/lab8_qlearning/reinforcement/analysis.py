# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    # If we make it impossible to fall off, then our value function does not
    # have to worry about the immense negative return. Thus, even from (1,1),
    # the agent will be compelled to move west; 10 is sufficiently greater than
    # 1 such that the high discount rate doesn't really do much. You would need
    # 22 steps to make the reward of going left even equal the reward of going
    # right since 0.9^(~22) = 0.1 so 0.9^22 * 10 ~ 1
    answerDiscount = 0.9
    answerNoise = 0
    return answerDiscount, answerNoise

def question3a():
    # If we make it impossible to accidentally fall off the cliff, then our
    # agent doesn't have to factor this possibility into its calculations.
    # Now, we set the discount rate to be super low so that we incentivize
    # leaving early
    answerDiscount = 0.001
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3b():
    # We want to incentivize leaving early by making the discount low enough
    # to where it makes the longer reward practically meaningless, choosing the
    # closer reward
    answerDiscount = 0.2
    # We want to have some noise factor so that the agent is averse to the cliff
    answerNoise = 0.1
    # I still don't think we need to worry about the reward yet, but in theory
    # we would want to make it 0 <= r < 1 so that it decays, deincentivizing
    # longer games
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    # We want the higher discount rate so that the agent will prefer the
    # distant reward over the closer one
    answerDiscount = 0.9
    # If we make it impossible to accidentally fall, the agent won't worry
    # about safety
    answerNoise = 0
    # Since the distant reward is so high we still don't have to worry about
    # living reward lol
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    # Again, high discount rate to incentivize further exit
    answerDiscount = 0.9
    # Decent noise rate to create fear of falling at every state
    answerNoise = 0.2
    # Still don't have to worry about living reward since falling fear is so
    # high they will be safe anyways lmao
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    # The discount rate being 0 neautralizes any exit point, positive or
    # negative
    answerDiscount = 0
    # Make accidental death impossible
    answerNoise = 0
    # Living reward is greater than exit, so agent will keep living
    answerLivingReward = 1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = None
    answerLearningRate = None
    # I tried many combinations. MANY combinations. I just don't think 50
    # episodes is enough to explore every option. If you have an epsilon of 1
    # so that the choices are completely random and a learning rate of 1 so that
    # new data is given priority, even then we don't reach the goal. In what I
    # conceptualized to be the most exploratory conditions, even then 50
    # episodes isn't enough to explore everything :(
    # Very angy about this wild goose chase for an answer >:^()
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
