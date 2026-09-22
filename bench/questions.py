#!/usr/bin/env python3
"""
50-question benchmark for the CPU JEV/System-One deployment.

Shape: every item is a System-One decision (state, question, typed options),
so the model never generates text. Mixed difficulty + mixed task types so the
score is comparable to what a full-size frontier model would get.

Tiers:
  T1 (1-15)  easy single-fact / sentiment / topic
  T2 (16-35) moderate reasoning, arithmetic, entailment, code, logic
  T3 (36-50) hard: multi-step, traps, world knowledge, tricky logic
"""
import json

# (id, tier, task_type, state, question, options, answer)
ITEMS = [
 # ---------------- T1: easy ----------------
 (1,1,"sentiment","The plot is thin, but the two leads are so charming that I left the cinema smiling.","What is the sentiment of this review?",["negative","positive"],"positive"),
 (2,1,"topic","Shares of the chipmaker jumped 8% after it raised its full-year revenue forecast.","Which news section does this belong to?",["World","Sports","Business","Science/Technology"],"Business"),
 (3,1,"math","What is 17 + 26?","What is the result?",["41","42","43","44"],"43"),
 (4,1,"math","What is 9 * 7?","What is the result?",["56","63","72","81"],"63"),
 (5,1,"capital","Consider the country France.","What is its capital city?",["Lyon","Marseille","Paris","Nice"],"Paris"),
 (6,1,"sentiment","The food was cold and the waiter was rude.","What is the sentiment of this review?",["negative","positive"],"negative"),
 (7,1,"entailment","Premise: A man is playing a guitar on stage.\nHypothesis: Someone is performing music.","Does the premise entail the hypothesis?",["yes","no"],"yes"),
 (8,1,"topic","The team scored a touchdown in the final seconds to win the championship.","Which section does this belong to?",["World","Sports","Business","Science/Technology"],"Sports"),
 (9,1,"math","What is 100 - 37?","What is the result?",["53","63","67","73"],"63"),
 (10,1,"sentiment","An absolute masterpiece; I have never seen anything like it.","What is the sentiment of this review?",["negative","positive"],"positive"),
 (11,1,"capital","Consider the country Japan.","What is its capital city?",["Kyoto","Osaka","Seoul","Tokyo"],"Tokyo"),
 (12,1,"math","What is 144 / 12?","What is the result?",["10","11","12","14"],"12"),
 (13,1,"entailment","Premise: The cat is sleeping on the sofa.\nHypothesis: The cat is awake.","Does the premise entail the hypothesis?",["yes","no"],"no"),
 (14,1,"topic","Researchers published a study on CRISPR gene editing in mice.","Which section does this belong to?",["World","Sports","Business","Science/Technology"],"Science/Technology"),
 (15,1,"math","What is 8 + 15?","What is the result?",["21","22","23","24"],"23"),

 # ---------------- T2: moderate ----------------
 (16,2,"math","What is 23 * 14?","What is the result?",["312","322","332","342"],"322"),
 (17,2,"logic","All roses are flowers. All flowers need water.\nTherefore:","Do roses need water?",["yes","no"],"yes"),
 (18,2,"math","A shirt costs 40 dollars. It is discounted by 25 percent.","What is the final price in dollars?",["28","30","32","35"],"30"),
 (19,2,"entailment","Premise: No reptiles have fur. A snake is a reptile.\nHypothesis: A snake has fur.","Does the premise entail the hypothesis?",["yes","no"],"no"),
 (20,2,"code","Consider this Python code:\nx = [1, 2, 3, 4]\ny = x[1:3]\nprint(y)","What does it print?",["[1, 2]","[2, 3]","[3, 4]","[2, 3, 4]"],"[2, 3]"),
 (21,2,"logic","If it rains, the ground gets wet. The ground is dry.","Did it rain?",["yes","no"],"no"),
 (22,2,"math","What is 15 percent of 240?","What is the result?",["24","30","36","40"],"36"),
 (23,2,"code","Consider this Python code:\nprint(len('hello'))","What does it print?",["4","5","6","Error"],"5"),
 (24,2,"entailment","Premise: The movie was two hours long and very boring.\nHypothesis: The movie was enjoyable.","Does the premise entail the hypothesis?",["yes","no"],"no"),
 (25,2,"math","A train travels 60 km/h for 2.5 hours.","How far does it travel in km?",["120","140","150","160"],"150"),
 (26,2,"logic","Some birds cannot fly. A penguin is a bird.","Can we conclude the penguin cannot fly?",["yes","no"],"no"),
 (27,2,"code","Consider this Python code:\nprint(2 ** 3)","What does it print?",["6","8","9","23"],"8"),
 (28,2,"math","What is 7 * 8 - 6?","What is the result?",["48","50","52","56"],"50"),
 (29,2,"logic","Every student who studies passes. Maria did not pass.","Did Maria study?",["yes","no"],"no"),
 (30,2,"code","Consider this Python code:\nd = {'a': 1, 'b': 2}\nprint(d.get('c', 0))","What does it print?",["0","1","2","None"],"0"),
 (31,2,"math","What is 1/4 expressed as a percentage?","What is the result?",["20 percent","25 percent","40 percent","50 percent"],"25 percent"),
 (32,2,"entailment","Premise: The keys are in the drawer.\nHypothesis: The keys are somewhere in the house.","Does the premise entail the hypothesis?",["yes","no"],"yes"),
 (33,2,"math","If 3x = 21, what is x?","What is the value?",["5","6","7","8"],"7"),
 (34,2,"logic","All squares are rectangles. This shape is a square.","Is this shape a rectangle?",["yes","no"],"yes"),
 (35,2,"code","Consider this Python code:\nprint(sorted([3,1,2]))","What does it print?",["[1, 2, 3]","[3, 2, 1]","[3, 1, 2]","Error"],"[1, 2, 3]"),

 # ---------------- T3: hard ----------------
 (36,3,"math","A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 dollar more than the ball.","How much does the ball cost in dollars?",["0.05","0.10","0.15","1.00"],"0.05"),
 (37,3,"logic","A farmer has 17 sheep. All but 9 run away.","How many sheep are left?",["8","9","17","0"],"9"),
 (38,3,"math","If 5 machines take 5 minutes to make 5 widgets, how long do 100 machines take to make 100 widgets?","How long in minutes?",["5","20","100","500"],"5"),
 (39,3,"knowledge","Consider the element with atomic number 79.","What is its chemical symbol?",["Ag","Au","Cu","Pt"],"Au"),
 (40,3,"math","In a lake, a patch of lily pads doubles in size every day. It covers the lake in 48 days.","How many days to cover half the lake?",["24","36","47","47.5"],"47"),
 (41,3,"knowledge","Consider the year 1969.","Which event happened that year?",["First human on the Moon","Fall of the Berlin Wall","End of World War II","Sinking of the Titanic"],"First human on the Moon"),
 (42,3,"logic","Three boxes: one has apples, one oranges, one both. All labels are wrong. You pick from the box labeled 'both' and get an apple.","What does the box labeled 'both' actually contain?",["apples only","oranges only","both","cannot be determined"],"apples only"),
 (43,3,"math","A book costs 20 dollars plus half its price.","What is the total price in dollars?",["30","40","50","60"],"40"),
 (44,3,"knowledge","Consider the largest planet in the solar system.","Which planet is it?",["Earth","Jupiter","Neptune","Saturn"],"Jupiter"),
 (45,3,"logic","Consider this syllogism: some A are B, and all B are C.","Which statement is certainly true?",["Some A are C","All A are C","No A are C","All C are A"],"Some A are C"),
 (46,3,"code","Consider this Python code:\nprint(bool('') or bool('x'))","What does it print?",["False","True","None","Error"],"True"),
 (47,3,"math","A car depreciates 20 percent per year. It is worth 10000 dollars now.","What is it worth after 2 years in dollars?",["6000","6400","7200","8000"],"6400"),
 (48,3,"knowledge","Consider the author of the play 'Hamlet'.","Who wrote it?",["Charles Dickens","Christopher Marlowe","John Milton","William Shakespeare"],"William Shakespeare"),
 (49,3,"logic","You have two ropes; each burns in exactly 60 minutes but unevenly.","Can you measure 45 minutes by lighting ropes?",["yes","no"],"yes"),
 (50,3,"math","What is the next number in the sequence 2, 6, 12, 20, 30?","What comes next?",["36","40","42","48"],"42"),
]

def load():
    return ITEMS

if __name__ == "__main__":
    print(f"{len(ITEMS)} items")
    from collections import Counter
    print("by tier:", dict(Counter(t for _,t,*_ in ITEMS)))
