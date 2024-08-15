class AI:
    def __init__(self,name,alpha,gamma,epsilon,epsilon_decay,epsilon_min):
        self.name=name
        self.alpha=alpha
        self.gamma=gamma
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.epsilon_min=epsilon_min

    def __str__(self):
        return f"Név: {self.name} (a:{self.alpha}, g:{self.gamma}, e:{self.epsilon}, ed:{self.epsilon_decay}, em:{self.epsilon_min})"

al = AI("Al", 0.1, 0.9, 0.1, 1, 0.1)
fred = AI("Fred", 0.1, 0.9, 1.0, 0.995, 0.01)
nick = AI("Nick", 0.2, 0.9, 0.5, 0.995, 0.0001)

ais = [al, fred, nick]

def ai_list():
    listed_ai = []
    for i, ai in enumerate(ais, start=1):
        listed_ai.append(f"{i} - {ai}")
    return listed_ai

def ai_select():
    ai_list()
    try:
        selection=int(input("Add meg a választott AI számát: "))
    except ValueError as v:
        selection=f"Érvénytelen érték!\n{v}"
    except Exception as e:
        selection=f"Hiba a futás közben!\n{e}"
    return selection

def ai_set():
    select = ai_select()
    if select is not int:
        selected_ai = select
    if select>=1 and select<=len(ais):
        selected_ai = ais[select - 1]
    else:
        selected_ai = "Érvénytelen választás!"
    return selected_ai
