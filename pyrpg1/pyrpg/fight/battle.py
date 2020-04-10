


def check_battle_input(player_inputs):
    if player_inputs == 'atk':
        return False
    elif player_inputs == 'item':
        return False
    elif player_inputs == 'run':
        return False
    elif player_inputs == None:
        print('need an input')
        return True
    else:
        print('error, repeat command')
        return True


# def command(player_input = None, attacker = None, defender = None):
    
#         if player_input == 'atk':
#             attack(attacker, defender)
#             return False

#         elif player_input == 'item':
#             return False

#         elif player_input == 'run':
#             return False

#         elif player_input == None:
#             print('need an input')
#             return True

#         else:
#             print('error, repeat command')
#             return True

def attack(attacker, defender):

    defender.hp -= 2
    print('{} damage {} for {} hp'.format(attacker.name, defender.name, 1), '\n')


def check(player, monster):

    if player.hp <= 0 :
        print(f'{player.name} is slain.', '\n')
        return False

    elif monster.hp <= 0:
        print(f'{monster.name} is defeated!', '\n')
        return False

    else:
        return True

def alive(player):

    if player.hp >= 0:
        return True

    else:
        return False

def check_player_death(player):

    if player.hp <= 0:
        print('Game Over')
        quit()

    else:
        return True


