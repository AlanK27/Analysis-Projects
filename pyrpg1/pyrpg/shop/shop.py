


class shop:

    def __init__(self):
        self.shop_list = ['shop list']

    def shoplist(self):
        print('Shop items:')
        shoplist = {    1:'herb',
                        2:'potion'
        }
        for n in shoplist.keys():
           print(n, shoplist[n])
        return shoplist

    def check_bag_size(self, player):
        if len(player.list1) >= 10:
            print('bag is full', '\n')
            return False
        elif len(player.list1) <= 0 :
            print('bag is empty', '\n')
            return False
        else:
            return True

    def buy(self, player, inputs):
        if inputs in list(self.shoplist().keys()):
            player.list1.append(inputs)
            print('bought {}'.format(player.dic()[inputs])
            return True

        elif inputs == 'quit':
            return False

        else:
            print('no such item')
            return False


    def sell(self, player, inputs):
        if inputs in range(len(player.list1)):
            itemm = player.list1.pop(inputs)
            if itemm in list(player.dic().keys()):
                print('Sold {} for {} gold'.format(player.dic()[itemm], 5))
                player.gold += 5
                return True
        
        elif input == 'quit':
            return False

        else:
            print('no such item')
            return False


       

    # def check_shop_input(self, input):
    #     if input in range(3):
    #         return True
    #     else:
    #         return False

    # def shop_input(input):
    #     shop = True
    #     while shop:
    #         if str(input).lower() == 'buy':
                
    #             shop = False
    #         elif str(input).lower() == 'sell':

    #             shop = False
    #         elif str(input).lower() == 'item':

    #             shop = False
    #         elif str(input).lower() == 'quit':

    #             shop = False
    #         else:
    #             print('Wrong input')



