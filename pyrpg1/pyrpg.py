# from pyrpg.enemy.enemy import monsters
from pyrpg.status.status import status
from pyrpg.script.script1 import script

def main():

    name = input('chacter name? \n')

    print('roll for your character stats:')
    x = input('roll')
    yesno = True
    while yesno:
        player = status(status.roll())
        player.print_stats()
        p_input = str(input('y/n   ')).lower()
        if p_input == 'y':
            player.name = name
            yesno = False
        elif p_input == 'quit':
            quit()
            break

    script(player)

if __name__ == '__main__':
    main()
    # script(status(status.roll()))
