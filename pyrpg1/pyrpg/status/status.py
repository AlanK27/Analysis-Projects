import random
from pyrpg.items.items import items
from pyrpg.shop.shop import shop

class status(items, shop):

    def __init__(  self, args, name = 'Edrik', lvl = 1, \
                    exp = 0, gold = 0):
        super().__init__()
        self.hpmax = args[0]
        self.hp = args[0]
        self.mpmax = args[1]
        self.mp = args[1]
        self.atk = args[2]
        self.defn = args[3]
        self.name = name
        self.lvl = lvl
        self.exp = exp
        self.gold = gold
        
    def roll(hp=6, hp1=8, mp=1, mp1=3,\
            atk=20, atk1=30, defn=10, defn1=15):
        hp = random.randrange(hp,hp1)
        mp = random.randrange(mp, mp1)
        atk = random.randrange(atk, atk1)
        defn = random.randrange(defn, defn1)
        return hp, mp, atk, defn

    def __str__(self):
        return " HP: {}   MP: {}".format(self.hp, self.mp)
        
        
    def print_stats(self):
        print('HP: ', self.hp, '\n'
            'MP: ', self.mp, '\n'
            'Attack: ', self.atk, '\n'
            'Defense: ', self.defn
            )
        
    def stats_increase(self):
        self.hpmax += random.randrange(2,4)
        self.mpmax += random.randrange(0,3)
        self.atk += random.randrange(1,4)
        self.defn += random.randrange(0,2)

        self.hp = self.hpmax
        self.mp = self.mpmax

        print(  'HP increased to ', self.hpmax, '\n'
                'MP increased to ', self.mpmax, '\n'
                'ATK increased to ', self.atk, '\n'
                'DEF increased to ', self.defn, '\n'
        )


    def __add__(self, others):
        self.exp += others.exp
        self.gold += others.gold













