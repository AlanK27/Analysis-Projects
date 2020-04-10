from pyrpg.status.status import status


class monsters:

    def monster_list():
        return ['slime', 'red slime']

    def slime():
        stats = status(status.roll(hp=1, hp1=2))
        stats.name = 'slime'
        stats.exp = 2
        stats.gold = 2
        return stats

    def red_slime():
        stats = status(status.roll(hp=4, hp1=7))
        stats.name = 'slime'
        stats.exp = 3
        stats.gold = 5
        return stats

    def other():
        print('this is a slime')
        







