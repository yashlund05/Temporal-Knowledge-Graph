"""
Populate CronQuestions dataset in data/raw/cronquestions.

Generates the entity mapping, temporal knowledge graph splits, and
temporal question answering splits according to the CronQuestions format.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import CRONQUESTIONS_DIR


def populate():
    CRONQUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    kg_dir = CRONQUESTIONS_DIR / "full_kg"
    kg_dir.mkdir(parents=True, exist_ok=True)
    q_dir = CRONQUESTIONS_DIR / "questions"
    q_dir.mkdir(parents=True, exist_ok=True)

    entities = {
        "Q30": "United States",
        "Q142": "France",
        "Q183": "Germany",
        "Q145": "United Kingdom",
        "Q159": "Russia",
        "Q148": "China",
        "Q668": "India",
        "Q17": "Japan",
        "Q38": "Italy",
        "Q16": "Canada",
        "Q76": "Barack Obama",
        "Q22686": "Donald Trump",
        "Q6279": "Joe Biden",
        "Q1124": "Bill Clinton",
        "Q207": "George W. Bush",
        "Q7747": "Vladimir Putin",
        "Q23530": "Dmitry Medvedev",
        "Q567": "Angela Merkel",
        "Q82872": "Olaf Scholz",
        "Q3052772": "Emmanuel Macron",
        "Q157": "François Hollande",
        "Q192": "Nicolas Sarkozy",
        "Q192": "David Cameron",
        "Q264766": "Theresa May",
        "Q180589": "Boris Johnson",
        "Q15031": "Xi Jinping",
        "Q15029": "Hu Jintao",
        "Q1058": "Narendra Modi",
        "Q208643": "Manmohan Singh",
        "Q132345": "Shinzo Abe",
        "Q39228": "Justin Trudeau",
        "Q7184": "NATO",
        "Q1065": "United Nations",
        "Q458": "European Union",
        "Q889": "Afghanistan",
        "Q79": "Egypt",
        "Q843": "Pakistan",
        "Q801": "Israel",
        "Q794": "Iran",
        "Q96": "Mexico",
        "Q155": "Brazil",
        "Q1060": "Kofi Annan",
        "Q1253": "Ban Ki-moon",
        "Q12664": "António Guterres",
    }

    # Write wd_id2entity_text.txt
    mapping_file = CRONQUESTIONS_DIR / "wd_id2entity_text.txt"
    with open(mapping_file, "w", encoding="utf-8") as f:
        for qid, name in entities.items():
            f.write(f"{qid}\t{name}\n")

    # Knowledge graph facts: subject_id, relation_id, object_id, start_time, end_time
    kg_facts = [
        # US Presidents
        ("Q1124", "head_of_government", "Q30", "1993", "2001"),
        ("Q207", "head_of_government", "Q30", "2001", "2009"),
        ("Q76", "head_of_government", "Q30", "2009", "2017"),
        ("Q22686", "head_of_government", "Q30", "2017", "2021"),
        ("Q6279", "head_of_government", "Q30", "2021", "2025"),
        # Russia Leaders
        ("Q7747", "head_of_government", "Q159", "2000", "2008"),
        ("Q23530", "head_of_government", "Q159", "2008", "2012"),
        ("Q7747", "head_of_government", "Q159", "2012", "####-##-##"),
        # Germany Chancellors
        ("Q567", "head_of_government", "Q183", "2005", "2021"),
        ("Q82872", "head_of_government", "Q183", "2021", "####-##-##"),
        # France Presidents
        ("Q192", "head_of_government", "Q142", "2007", "2012"),
        ("Q157", "head_of_government", "Q142", "2012", "2017"),
        ("Q3052772", "head_of_government", "Q142", "2017", "####-##-##"),
        # UK Prime Ministers
        ("Q192", "head_of_government", "Q145", "2010", "2016"),
        ("Q264766", "head_of_government", "Q145", "2016", "2019"),
        ("Q180589", "head_of_government", "Q145", "2019", "2022"),
        # China
        ("Q15029", "head_of_government", "Q148", "2003", "2013"),
        ("Q15031", "head_of_government", "Q148", "2013", "####-##-##"),
        # India
        ("Q208643", "head_of_government", "Q668", "2004", "2014"),
        ("Q1058", "head_of_government", "Q668", "2014", "####-##-##"),
        # UN Secretary-General
        ("Q1060", "head_of_government", "Q1065", "1997", "2006"),
        ("Q1253", "head_of_government", "Q1065", "2007", "2016"),
        ("Q12664", "head_of_government", "Q1065", "2017", "####-##-##"),
        # International Memberships
        ("Q30", "member_of", "Q7184", "1949", "####-##-##"),
        ("Q145", "member_of", "Q7184", "1949", "####-##-##"),
        ("Q142", "member_of", "Q7184", "1949", "####-##-##"),
        ("Q183", "member_of", "Q7184", "1955", "####-##-##"),
        ("Q142", "member_of", "Q458", "1958", "####-##-##"),
        ("Q183", "member_of", "Q458", "1958", "####-##-##"),
        ("Q145", "member_of", "Q458", "1973", "2020"),
    ]

    # Split into train/valid/test
    n = len(kg_facts)
    train_split = kg_facts[: int(n * 0.7)]
    valid_split = kg_facts[int(n * 0.7): int(n * 0.85)]
    test_split = kg_facts[int(n * 0.85):]

    for split_name, facts in [("train", train_split), ("valid", valid_split), ("test", test_split)]:
        path = kg_dir / f"{split_name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for s, r, o, st, et in facts:
                f.write(f"{s}\t{r}\t{o}\t{st}\t{et}\n")

    # Build representative CronQuestions
    questions_data = [
        {
            "question": "Who was the head of government of United States in 2012?",
            "answer": ["Barack Obama"],
            "type": "time:entity",
            "entities": ["United States"],
            "timestamps": ["2012"],
        },
        {
            "question": "Who was the head of government of United States in 2018?",
            "answer": ["Donald Trump"],
            "type": "time:entity",
            "entities": ["United States"],
            "timestamps": ["2018"],
        },
        {
            "question": "Who was the head of government of United States in 2022?",
            "answer": ["Joe Biden"],
            "type": "time:entity",
            "entities": ["United States"],
            "timestamps": ["2022"],
        },
        {
            "question": "Who was the head of government of Germany in 2015?",
            "answer": ["Angela Merkel"],
            "type": "time:entity",
            "entities": ["Germany"],
            "timestamps": ["2015"],
        },
        {
            "question": "Who was the head of government of France in 2014?",
            "answer": ["François Hollande"],
            "type": "time:entity",
            "entities": ["France"],
            "timestamps": ["2014"],
        },
        {
            "question": "Who was the head of government of France in 2019?",
            "answer": ["Emmanuel Macron"],
            "type": "time:entity",
            "entities": ["France"],
            "timestamps": ["2019"],
        },
        {
            "question": "Who was the head of government of Russia in 2010?",
            "answer": ["Dmitry Medvedev"],
            "type": "time:entity",
            "entities": ["Russia"],
            "timestamps": ["2010"],
        },
        {
            "question": "Who was the head of government of Russia in 2016?",
            "answer": ["Vladimir Putin"],
            "type": "time:entity",
            "entities": ["Russia"],
            "timestamps": ["2016"],
        },
        {
            "question": "Who was the head of government of India in 2010?",
            "answer": ["Manmohan Singh"],
            "type": "time:entity",
            "entities": ["India"],
            "timestamps": ["2010"],
        },
        {
            "question": "Who was the head of government of India in 2018?",
            "answer": ["Narendra Modi"],
            "type": "time:entity",
            "entities": ["India"],
            "timestamps": ["2018"],
        },
        {
            "question": "Who was the head of government of China in 2008?",
            "answer": ["Hu Jintao"],
            "type": "time:entity",
            "entities": ["China"],
            "timestamps": ["2008"],
        },
        {
            "question": "Who was the head of government of China in 2016?",
            "answer": ["Xi Jinping"],
            "type": "time:entity",
            "entities": ["China"],
            "timestamps": ["2016"],
        },
        {
            "question": "Who was the head of government of United Kingdom in 2017?",
            "answer": ["Theresa May"],
            "type": "time:entity",
            "entities": ["United Kingdom"],
            "timestamps": ["2017"],
        },
        {
            "question": "Who was the head of government of United Kingdom in 2020?",
            "answer": ["Boris Johnson"],
            "type": "time:entity",
            "entities": ["United Kingdom"],
            "timestamps": ["2020"],
        },
        {
            "question": "Who was the head of government of United Nations in 2014?",
            "answer": ["Ban Ki-moon"],
            "type": "time:entity",
            "entities": ["United Nations"],
            "timestamps": ["2014"],
        },
        {
            "question": "Who was the head of government of United Nations in 2019?",
            "answer": ["António Guterres"],
            "type": "time:entity",
            "entities": ["United Nations"],
            "timestamps": ["2019"],
        },
        {
            "question": "Who served as head of government of United States before Donald Trump?",
            "answer": ["Barack Obama"],
            "type": "before",
            "entities": ["United States", "Donald Trump"],
            "timestamps": ["2017"],
        },
        {
            "question": "Who served as head of government of United States after Donald Trump?",
            "answer": ["Joe Biden"],
            "type": "after",
            "entities": ["United States", "Donald Trump"],
            "timestamps": ["2021"],
        },
        {
            "question": "Who was head of government of United States when Angela Merkel was Chancellor in 2014?",
            "answer": ["Barack Obama"],
            "type": "simultaneous",
            "entities": ["United States", "Angela Merkel"],
            "timestamps": ["2014"],
        },
        {
            "question": "When did Barack Obama serve as head of government of United States?",
            "answer": ["2009", "2017"],
            "type": "entity:time",
            "entities": ["Barack Obama", "United States"],
            "timestamps": ["2009", "2017"],
        },
    ]

    # Save to questions/train.json, valid.json, test.json
    for split_name in ["train", "valid", "test"]:
        path = q_dir / f"{split_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(questions_data, f, indent=2)

    print(f"CronQuestions successfully populated at {CRONQUESTIONS_DIR}")
    print(f"Entities: {len(entities)}, Facts: {len(kg_facts)}, Questions: {len(questions_data)}")


if __name__ == "__main__":
    populate()
