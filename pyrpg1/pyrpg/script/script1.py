from random import randint
from pyrpg.status.status import status
from pyrpg.enemy.enemy import monsters
from pyrpg.fight.battle import check_battle_input, check, alive, check_player_death, attack
from pyrpg.items.items import items
from pyrpg.shop.shop import shop


def script(player):
    while alive(player):
        # for i in range(2):
        #     print('slime appears! Command?', '\n')
        #     fight_start(player)
        rest_command(player)
    print('Game Over')


def fight_start(player):

    monster = monsters.slime()
    fighting = True
    use = False

    #   start fight
    while fighting:
        use = False
        print(player, '\n')
        # player input command
        player_input = input('atk, item, run     ')
        while check_battle_input(player_input):
            player_input = input('atk, item, run     ')

        # using item
        while not(use):
            if player_input == 'item':
                player.check_item()
                use = player.use_item()
                if not(use):
                    player_input = input('atk, item, run     ')
                    while check_battle_input(player_input):
                        player_input = input('atk, item, run     ')
            else:
                use = True

        # attack
        if player_input == 'atk':
            attack(attacker = player, defender = monster)

        # fleeing
        if player_input == 'run':
            if randint(0,4) == 0:
                print('you fled', '\n')
                break
            else:
                print('cannot escape', '\n')

        # monster attack
        attack(attacker = monster, defender = player)

        # check death marker
        check_player_death(player)
        fighting = alive(monster)
       
    if not(fighting): 
        print('you gained {} exp'.format(monster.exp), '\n')
        player + monster
        if player.exp%10  == 0:
            player.stats_increase()
        

def rest_command(player):

    rest_count = True
    shopping = True

    while rest_count:
        print('would you like to:', '\n', 'stats, rest, shop, item, continue', '\n')
        p_input = input()

        if p_input == 'rest':
            player.hp = player.hpmax
            print('rested')
            player.print_stats()
            input()
            tick = False

        elif p_input == 'item':
            player.check_item()
        
        elif p_input == 'stats':
            player.print_stats()

        elif p_input == 'shop':
            player.shoplist()
            shopping = True
            
            while shopping:
                print('Buy, Sell, bag, quit')
                shop_input = input('\n')

                if str(shop_input).lower() == 'buy':

                    if player.check_bag_size(player):
                        shop_in = False

                        while not(shop_in):
                            player.shoplist()
                            shop_in = input('which item buy')
                    
                            if shop_in.isnumeric():
                                shop_in = player.buy(player, int(shop_in))
                                
                            else:
                                print('error, input again')

                elif str(shop_input).lower() == 'sell':

                    if player.check_bag_size(player):
                        sell_in = False

                        while not(sell_in):
                            player.check_item()
                            sell_in = input('which item to sell')
                            
                            if sell_in.isnumeric():
                                sell_in = player.sell(player, int(sell_in))




                elif str(shop_input).lower() == 'bag':
                    player.check_item()

                elif str(shop_input).lower() == 'quit':
                    shopping = False

                else:
                    print('error, input again', '\n')

        else:
            print('error, repeat command')


