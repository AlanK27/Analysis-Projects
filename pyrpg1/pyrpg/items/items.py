from pyrpg.items.item_lists.item_lists import item_list

class items(item_list):
    
    
    def __init__(self, list1 = [1,1,2] ):
        super().__init__()
        self.list1 = list1

    def check_item(self):
  
        for a,b in enumerate(list(map(self.dic().get, self.list1))):
            print(a,b)
        

    def use_item(self):

        ins = input('Use Item? y/n')
        if str(ins.lower()) == 'y':
            print('\n')
            item_num = int(input('Which item number?'))
            item_index = self.list1.pop(self.list1[item_num])
            self.using_item(item_index)
            return True

        elif str(ins.lower()) == 'n':
            return False

        else:
            print('error, input again', '\n')
            return False


    def using_item(self, item_nu):
        if item_nu == 1:
            self.hp += 30
            print(self.hp)
            if self.hp > self.hpmax:
                self.hp = self.hpmax

        if item_nu == 2:
            self.mp += 15
            if self.mp > self.mpmax:
                self.mp = self.mpmax

        if item_nu == 3:
            pass
