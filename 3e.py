def finite_horizon_bokoboko(word_length):
    states_dict = {
        'B': {
            word_length-1: (1,'|'),

        },
        'K': {
            word_length-1: (1,'|'),
        },
        'O': {
            word_length-1: (1,'|'),
        }
    }
    actions_dict_source_dest = {
        'B': {
            'B': 1 / 0.148,
            'K': 1 / 0.481,
            'O': 1 / 0.371
        },
        'K': {
            'B': 1 / 0.5,
            'K': 1,
            'O': 1 / 0.5
        },
        'O': {
            'B': 1 / 0.33333,
            'K': 1/ 0.33333,
            'O': 1/ 0.33333
        }
    }

    for i in range(word_length - 2, -1, -1):
        for source in states_dict.keys():
          min_action_value = None
          min_action_dest = None
          for dest in states_dict.keys():
            action_cost = actions_dict_source_dest[source][dest]
            next_state_cost_to_end = states_dict[dest][i+1][0]
            source_v_compute = action_cost * next_state_cost_to_end;
            if(not min_action_dest):
              min_action_value = source_v_compute
              min_action_dest = dest
            if(source_v_compute < min_action_value):
              min_action_value = source_v_compute
              min_action_dest = dest
          states_dict[source][i] = (min_action_value,min_action_dest)
    next_node = 'B'
    print('B', "->")
    word = 'B'
    for i in range(word_length):
      print(states_dict[next_node][i][1], "->")
      next_node = states_dict[next_node][i][1]
      word = word + next_node
    print("Final word: '{}'".format(word[:word_length]))




finite_horizon_bokoboko(5)
