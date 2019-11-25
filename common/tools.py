import re

measures = dict()
measures['dash'] = 0.462
measures['dashes'] = 0.462
measures['tsp'] = 4.929
measures['teaspoon'] = 4.929
measures['top'] = 10
measures['drops'] = 1
measures['microdrops'] = 0.5
measures['splash'] = 1
measures['scoop'] = 70
measures['cl'] = 70
measures['float'] = 1
measures['ml'] = 1
measures['oz'] = 30
measures['ounce'] = 30
measures['ounces'] = 30
measures['c'] = 30
measures['inch'] = 5
measures['pinch'] = 5
measures['cup'] = 250
measures['cups'] = 250
measures['tbsp'] = 15
measures['packages'] = 100
measures['tincture'] = 20
measures['rinse'] = 5
measures['bottle'] = 333
measures['bottles'] = 333
measures['spoon'] = 15
measures['bsp'] = 15
measures['stick'] = 10
measures['bag'] = 200
measures['strip'] = 10
measures['whole'] = 100
measures['packet'] = 100
measures['tblsp'] = 15
measures['shot'] = 50
measures['shots'] = 50
measures['part'] = 50
measures['parts'] = 50
measures['drop'] = 1
measures['full glass'] = 250
measures['pint'] = 568
measures['l'] = 1000
measures['jigger'] = 44
measures['jiggers'] = 44
measures['gr'] = 1
measures['to taste'] = 25
measures['qt'] = 946.3
measures['wedge'] = 20
measures['wedges'] = 20
measures['by taste'] = 5
measures['fresh'] = 50
measures['cubes'] = 50
measures['cube'] = 50
measures['oz bacardi'] = 30
measures['fill with'] = 100
measures['measure'] = 50
measures['measures'] = 50
measures['barspoon'] = 5

replacing_dict = {'1 1/4':'1.25', '1/4':'0.25', '1/3':'0.333', '1/6':'0.17', '1/2':'0.5', '1 1/2':'1.5', '2/3':'0.66', '1 3/4':'1.75' ,'3/4':'0.75', '2/5':'0.4', '1 fifth':'0.2',
                  '1/5':'0.2', '1-3':'2', '4-6':'5', '1/8':'0.125', '1 quart':'0.25', '2-4':'3', '3-4':'3.5', '2 or 3':'2.5', '4-5':'4.5', '1 0.5':'1.5',
                  '½':'0.5', '4 fifth':'0.8', '0.5 or 1':'0.75', '5 or 6':'5.5', '1 (1-inch)':'50', '1-2':'1.5', '2-3': '2.5', '5-6':'5.5'}

tmp = dict()
for key in sorted(measures.keys(), key=len, reverse=True):
    tmp[key] = measures[key]

measures = tmp

def formatMeasureValues(df, replacing_dict):
    for k in replacing_dict:
        df = df.replace({k:replacing_dict[k]}, regex=True)
    return df

def getNumber(val):
    if re.match(r'^-?\d+(?:\.\d+)?$', val) is None:
        return 0
    return float(val)

def findMeasure(ingr, measures):
    for m in measures:
        if m in ingr:
            return m
    return 'ounce'

def uncodeDecode(str):
    return str.encode('utf-8').decode('utf-8').replace('\xa0','').replace('\u200b','').replace('()', '').strip()
