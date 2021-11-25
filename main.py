# Jerry Xu
# CS 4375 Fall 2021 Homework 3 Part 1 (Value Iteration w/ Bellman's Equation)
import collections
import sys


def next_J_value(state, gamma, actions, rewards, table):
    """
    calculates and returns J^(k+1)(s_i) using Bellman's Equation
    also returns the action that resulted in the highest J^(k+1)(s_i)
    """
    poss_actions = actions[state]

    # get expected rewards for each action
    exp_rewards = []

    for action, next_states in poss_actions.items():
        exp_reward = sum([table[-1][1][next_state][1] * trans_prob for next_state, trans_prob in next_states.items()])
        exp_rewards.append((action, exp_reward))

    best_action, max_reward = max(exp_rewards, key=lambda x: x[1])

    return best_action, rewards[state] + (gamma * max_reward)


def value_iteration(gamma, actions, rewards, table, num_iterations):
    """
    does value iteration
    """
    for _ in range(num_iterations):
        line_number, J_values = table[-1]

        next_row = {}
        for state, J_val in J_values.items():
            next_row[state] = next_J_value(state, gamma, actions, rewards, table)
        table.append([line_number + 1, next_row])


def print_table(table):
    """
    prints the DP table to screen
    """
    for row in table:
        line_number, J_values = row
        print("After iteration {}:".format(line_number))
        jv = ""
        for state in J_values:
            jv += "({} {} {:.4f}) ".format(state, J_values[state][0], J_values[state][1])
        print(jv)


def main():
    # command line args
    num_states = sys.argv[1]  # unused
    num_actions = sys.argv[2]  # unused
    input_file = sys.argv[3]
    gamma = float(sys.argv[4])

    # read the input file
    with open(input_file) as f:
        input_dataset = [line.replace("(", "").replace(")", "").split() for line in f if line.strip()]

    # dictionary where k = state, v = the state's reward
    rewards = {state[0]: float(state[1]) for state in input_dataset}

    # dictionary containing the actions and transition probabilities at each state
    actions = {}
    for state in input_dataset:
        action = collections.defaultdict(dict)
        for i in range(2, len(state), 3):
            action[state[i]].update({state[i + 1]: float(state[i + 2])})
        actions[state[0]] = action

    # DP table
    table = [[1, {state[0]: (state[2], float(state[1])) for state in input_dataset}]]

    # fill out and print the DP table
    value_iteration(gamma, actions, rewards, table, 19)
    print_table(table)


if __name__ == "__main__":
    main()
