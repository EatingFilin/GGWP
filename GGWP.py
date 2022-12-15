import random
import string


import requests
import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET

from sklearn.neural_network import MLPRegressor


def execute():
    params = {
        "date_req1": (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%d/%m/%Y"),
        "date_req2": datetime.datetime.now().strftime("%d/%m/%Y"),
        "VAL_NM_RQ": "R01375"
    }
    response = requests.get('http://www.cbr.ru/scripts/XML_dynamic.asp', params=params)
    response_xml = response.text
    rates = ET.fromstring(response_xml)
    values = []
    tb = []

    for record in rates.findall("Record"):
        date = record.attrib["Date"]
        value = record.find("Value").text

        value = float(value.replace(",", ".")) / 10
        date = datetime.datetime.strptime(date, "%d.%m.%Y")

        values.append(value)
        tb.append([date, value, "actualno"])
    trange = 14
    tb1 = []
    past_col = []
    futr_col = []

    for i in range(trange, len(values) - trange):
        t = (values[(i - trange):(i + trange)])
        tb1.append(list(t))

    for i in range(trange):
        past_col.append(f'past_{i}')
        futr_col.append(f'futr_{i}')

    learndata = pd.DataFrame(tb1, columns=(past_col + futr_col))

    k = 19
    X = learndata[past_col][:-k]
    Y = learndata[futr_col][:-k]
    Xt = learndata[past_col][-k:]
    Yt = learndata[futr_col][-k:]

    MLP = MLPRegressor(hidden_layer_sizes=(113, 90, 400), max_iter=999, random_state=84)
    MLP.fit(X, Y)

    predicted = MLP.predict(Xt)

    i = 1
    for future in predicted[0]:
        tb.append([datetime.datetime.now() + datetime.timedelta(days=i), future, "predugadano"])
        i = i + 1
    data = pd.DataFrame(tb, columns=("date", "value", "aktualnost"))

    imgpath = draw(data)
    return imgpath



def draw(data):
    sns.set_theme(style="darkgrid")
    f, axes = plt.subplots(figsize=(20, 10))
    ax = sns.lineplot(x="date", y="value", data=data, hue="aktualnost", palette="Set2", ax=axes)
    ax.set_title("Exchange rates")
    name = "static/" + id_generator(48) + ".png"
    plt.savefig(name)
    return name


def id_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))