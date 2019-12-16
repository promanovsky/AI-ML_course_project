import re
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error,precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

"""
Общеиспользуемые структуры (словари), методы используемые в других скриптах
"""

sns.set()

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

def draw_confusion_matrix(y_true, y_pred, title):
    cm=confusion_matrix(y_true, y_pred)
    # print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(title=title,
           ylabel='True rating',
           xlabel='Predicted rating', )

    labels = ['2', '2.5', '3', '3.5', '4', '4.5', '5']
    if cm.shape[1] > 7:
        labels = ['  '] + labels
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)

    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    #plt.legend(('', '', ''), loc='upper right')

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

def show_classification_statistics(Y_Test, prediction, lenc, title):
    Y_Test = lenc.inverse_transform(Y_Test)
    prediction = lenc.inverse_transform(prediction)
    Y_Test = Y_Test.astype(np.str)
    prediction = prediction.astype(np.str)
    draw_confusion_matrix(Y_Test, prediction, title)
    print(classification_report(Y_Test,prediction))

def forest_classification_test(X, Y):
    lenc = LabelEncoder()
    Y = lenc.fit_transform(Y)
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 101)
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=500, verbose=1).fit(X_Train,Y_Train)
    print('Random forest classifier test training time =', time.process_time() - start)
    prediction = trainedforest.predict(X_Test)
    show_classification_statistics(Y_Test, prediction, lenc, 'RandomForestClassifier')

def tree_classification_test(X, Y):
    lenc = LabelEncoder()
    Y = lenc.fit_transform(Y)
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 101)
    start = time.process_time()
    trainedtree = DecisionTreeClassifier().fit(X_Train,Y_Train)
    print('Decision Tree classifier test training time =', time.process_time() - start)
    prediction = trainedtree.predict(X_Test)
    show_classification_statistics(Y_Test, prediction, lenc, 'DecisionTreeClassifier')

def gradient_boosting_classification_test(X, Y):
    lenc = LabelEncoder()
    Y = lenc.fit_transform(Y)
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 101)
    start = time.process_time()
    trainedforest = GradientBoostingClassifier(n_estimators=500, verbose=1).fit(X_Train,Y_Train)
    print('Gradient boosting classifier test training time =', time.process_time() - start)
    prediction = trainedforest.predict(X_Test)
    show_classification_statistics(Y_Test, prediction, lenc, 'GradientBoostingClassifier')

def forest_regression_test(X, Y):
    lenc = LabelEncoder()
    Y = lenc.fit_transform(Y)
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 101)
    start = time.process_time()
    trainedforest = RandomForestRegressor(n_estimators=500, verbose=1).fit(X_Train,Y_Train)
    print('Random forest regression test training time =', time.process_time() - start)
    predictionforest = trainedforest.predict(X_Test)
    print('RandomForestRegressor mean_squared_error',mean_squared_error(Y_Test,predictionforest))
    print("RandomForestRegressor precision = {}".format(precision_score(Y_Test, predictionforest.round(), average='weighted')))
    print("RandomForestRegressor recall = {}".format(recall_score(Y_Test, predictionforest.round(), average='weighted')))
    print("RandomForestRegressor accuracy = {}".format(accuracy_score(Y_Test, predictionforest.round())))

def gradient_boosting_regression_test(X, Y):
    lenc = LabelEncoder()
    Y = lenc.fit_transform(Y)
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 101)
    start = time.process_time()
    trainedforest = GradientBoostingRegressor(n_estimators=500, verbose=1).fit(X_Train,Y_Train)
    print('GradientBoostingRegressor test training time =', time.process_time() - start)
    predictionforest = trainedforest.predict(X_Test)
    print('GradientBoostingRegressor mean_squared_error',mean_squared_error(Y_Test,predictionforest))
    print("GradientBoostingRegressor precision = {}".format(precision_score(Y_Test, predictionforest.round(), average='weighted')))
    print("GradientBoostingRegressor recall = {}".format(recall_score(Y_Test, predictionforest.round(), average='weighted')))
    print("GradientBoostingRegressor accuracy = {}".format(accuracy_score(Y_Test, predictionforest.round())))

def showXGBTrainImportance(data, targetColumn, feature_columns, needSave=False):
    xgbTrainData = xgb.DMatrix(data, targetColumn, feature_names=feature_columns)
    param = {'max_depth':7, 'objective':'reg:linear', 'eta':0.2}
    model = xgb.train(param, xgbTrainData, num_boost_round=300)
    xgb.plot_importance(model, grid ="false", max_num_features=30, height=0.5)
    if needSave:
        plt.savefig('feature importance param'+str(np.random.randint(0, 100))+'.pdf',size=1024, format='pdf',bbox_inches="tight")
    plt.show()

ingredients_transformation = dict()
ingredients_transformation['wine_ingr'] = ['maurin quina','rose wine','sparkling wine','barsol perfecto amor (aperitif wine)','wine','rosé wine','moscato wine','white wine','port wine','white port', 'port', 'ruby port','tawny port',
                                'lbv port', 'red wine','lillet blanc','lillet rose','lillet rouge','champagne','champagne or prosecco','chilled champagne','madeira','sercial madeira','lustau east india solera','dry riesling','cardamaro',
                                'prosecco','sparkling rose','quinquina','pimm\'s no.', 'dubonnet rouge','dubonnet']
ingredients_transformation['scotch_ingr'] = ['scotch','islay scotch', 'single malt scotch', 'blended scotch', 'butterscotch schnapps', 'scotch whiskey', 'scotch whisky']
ingredients_transformation['whisky_ingr'] = ['whisky', 'nikka coffey grain whisky', 'nikka whisky taketsuru', 'j.h. cutter whisky', 'canadian whisky', 'johnnie walker','jim beam','jack daniel\'s','jack daniels','rye whiskey','whiskey','tennessee whiskey',
                                  'blended whiskey','irish whiskey', 'white whiskey','bourbon whiskey','whiskey barrel bitters','canadian whiskey','corn whiskey', 'bourbon', 'bourbon rye','old forester bourbon',
                                  'hirsch small batch reserve bourbon','genever', 'bols genever','prata cachaca','cachaca','glenrothes vintage reserve','everclear','rittenhouse rye','benriach curiositas single malt 10 yo',
                                  'yukon jack','grain alcohol','kirschwasser','crown royal','wild turkey','rock and rye']
ingredients_transformation['liquer_ingr'] = ['liquer','luxardo aperitivo', 'luxardo bitter', 'luxardo amaro abano','luxardo espresso italian liqueur','luxardo fernet', 'cointreau', 'luxardo amaretto di saschira', 'luxardo limoncello', 'strega',
                                  'tuaca', 'orange liqueur', 'coconut liqueur', 'coffee liqueur', 'herbal liqueur', 'elderflower liqueur', 'liqueur de violette', 'walnut liqueur', 'chocolate liqueur', 'ginger liqueur',
                                  'pomegranate liqueur', 'cherry liqueur', 'strawberry liqueur', 'raspberry liqueur', 'godiva liqueur', 'banana liqueur', 'maraschino liqueur', 'melon liqueur', 'honey liqueur', 'blackberry liqueur',
                                  'vanilla liqueur', 'pear liqueur','hot shot tropical fruit liqueur','lychee liqueur','peach liqueur','hazelnut liqueur', 'hpnotiq liqueur', 'apricot liqueur', 'chambord liqueur', 'anise liqueur',
                                  'blueberry liqueur', 'x-rated fusion liqueur', 'cream liqueur', 'pumpkin liqueur', 'ty ku liqueur', 'galliano liqueur', 'apple liqueur', 'curaçao liqueur', 'sambuca liqueur', 'amaro ciociaro liqueur',
                                  'amaretto liqueur', 'benedictine liqueur','amaro', 'averna amaro', 'ramazzotti amaro', 'amaro nonino', 'amaro montenegro', 'amaro meletti', 'triple sec','dry curacao', 'blue curacao', 'orange curacao',
                                  'the genuine curacao liqueur clear','curacao','benedictine','benedictine dom','bonal','campari','giffard pamplemousse','aperol','averna','ancho reyes','giffard banane du brésil','cynar','licor 43',
                                  'the king\'s ginger','anisette','old mr. boston anisette','aquavit','jägermeister','tia maria','hot damn','galliano','galliano l\'autentico','galliano l\'authentico','allspice dram',
                                  'cherry heering','frangelico','kahlua','t. germain','sambuca','pisang ambon','amaretto','amaretto di saronno','ouzo','kummel','old mr. boston kummel','advocaat','limoncello','swedish punch',
                                  'margarita mix','crème de menthe','crème de noyau','crème de violette','crème de fraise','crème de framboise','crème de mûre','crème de pêche','crème de cassis','outhern comfort',
                                  'st germain','kümmel','suze','ramazzotti','pimento dram','blue curaçao','orange curaçao','swedish flaggpunsch', 'mandarine napoleon']
ingredients_transformation['pepper_ingr'] = ['pepper','pepper corns', 'black pepper', 'pepper', 'cayenne pepper', 'pepper and salt', 'habanero pepper', 'green bell pepper', 'serrano pepper']
ingredients_transformation['syrup_ingr'] = ['syrup', 'monin cane syrup', 'pomegranate syrup', 'small hand foods gum syrup', 'cardamom syrup','agave syrup', 'demerara syrup','gomme syrup','ginger syrup','maple syrup','lavender syrup',
                                 'hibiscus syrup','pineapple syrup','rhubarb syrup','orgeat syrup','pear syrup','honey syrup','raspberry syrup','mint syrup','chocolate syrup','corn syrup','chile syrup','imple syrup',
                                 'spiced syrup','blackberry syrup','lapsong souchang syrup','basil syrup','toffee syrup','cherry syrup','trawberry syrup','rose syrup','elderflower syrup','gooseberry syrup', 'grenadine',
                                 'cherry grenadine', 'orgeat', 'small hand foods orgeat', 'house-made orgeat', 'velvet falernum','falernum','maraschino cherry','maraschino', 'coriander syrup', 'almond syrup']
ingredients_transformation['soda_ingr'] = ['soda', 'fever-tree soda','soda water','club soda','grape soda','lime soda','liter ub soda','7-up soda']
ingredients_transformation['vermouth_ingr'] = ['vermouth', 'tempus fugit alessio vermouth di torino rosso','tempus fugit alessio vermouth chinato', 'blanc vermouth', 'dry vermouth', 'dolin vermouth', 'red vermouth','cocchi vermouth di torino',
                                    'dryvermouth','weet vermouth','bianco vermouth','carpano antica','punt e mes']
ingredients_transformation['cinnamon_ingr'] = ['cinnamon', 'cinnamon tincture','cannella cinnamon cordial', 'ground cinnamon']
ingredients_transformation['brandy_ingr'] = ['brandy', 'calvados', 'peach brandy', 'apple brandy', 'brandy de jerez', 'apricot brandy', 'c. drouin calvados selection', 'cherry brandy', 'pear brandy', 'blackberry brandy', 'coffee brandy',
                                  'old mr. boston five star brandy', 'mr. boston five star brandy', 'cider brandy', 'plum brandy','c. drouin pommeau de normandie', 'pisco', 'barsol pisco','applejack']
ingredients_transformation['orange_ingr'] = ['orange', 'orange cream citrate', 'orange flower water','orange blossom water', 'orange juice', 'orange marmalade', 'orange peel', 'mandarin orange', 'orange sorbet', 'orange bitters',
                                  'orange spiral', 'orange zest','orange twist','orange wheel','orange wedge','oranges','orange slice','orange (cut into )','orange slie']
ingredients_transformation['absinthe_ingr'] = ['absinthe', 'vieux pontarlier absinthe francaise superieure', 'duplais swiss absinthe verte', 'absinthe or pastis', 'absinthe substitute', 'absinthe bitters', 'lucid absinthe', 'pernod absinthe']
ingredients_transformation['vodka_ingr'] = ['vodka','orange vodka','coconut vodka', 'vodka (or tequila)', 'lemon vodka','junmai sake','hangar 1 vodka','hophead vodka','grapefruit flavored vodka','pumpkin vodka','karlsson\'s gold vodka',
                                 'absolut vodka','vanilla vodka','peach vodka','cranberry vodka','raspberry vodka','cherry vodka','citrus vodka','green apple vodka','ml  vodka','espresso vodka','cake vodka',
                                 'pear vodka','pomegranate vodka (van gogh)','. vodka','iter vodka','stolichnaya vodka','chocolate vodka','blueberry vodka','ginger vodka','vodka (skyy)','black vodka','of vodka',
                                 'plain vodka','goldschlager','absolut citron','absolut kurant','peach schnapps','peppermint schnapps','apple schnapps','rumple minze','sake']
ingredients_transformation['sherry_ingr'] = ['sherry', 'oloroso sherry', 'manzanilla sherry','amontillado sherry','pedro ximenez sherry','15 yo sherry','fino sherry','cream sherry','dry sherry','sweet sherry','palo cortado sherry','pedro ximénez sherry',
                                  'olorosso sherry']
ingredients_transformation['beer_ingr'] = ['beer','lager', 'ginger beer', 'root beer','mexican beer','light beer','pale ale beer', 'ginger ale', 'ale','guinness stout','guinness','corona','chilled stout','raspberry lambic']
ingredients_transformation['gin_ingr'] = ['gin', 'genevieve gin', 'junipero gin', 'barr hill gin', 'plymouth gin', 'aviation gin', 'citadelle gin', 'old tom gin', 'sloe gin', 'dry gin', 'mr. boston gin', 'mint-flavored gin',
                               'tanqueray gin', 'bombay sapphire gin', 'gin or vodka', 'beefeater gin', 'american gin', 'premium gin' , 'bulldog gin', 'gin (new amsterdam gin)', 'gin (hendrick\'s gin)']
ingredients_transformation['lemonade_ingr'] = ['lemonade', 'fever-tree lemonade','good quality sharp lemonade', 'pink lemonade','kool-aid','cola','coca-cola','pepsi cola','7-up','mountain dew','zima','schweppes russchian','sprite','fruit punch', '7-up soda']
ingredients_transformation['bitter_ingr'] = ['bitter','chocolate bitters', 'miracle mile forbidden barrel aged bitters', 'cranberry bitters', 'dr. adam\'s boker\'s bitters', 'grapefruit bitters', 'tempus fugit abbott\'s bitters',
                                  'peach bitters', 'cherry bitters','celery bitters','jerry thomas bitters','bitters','angostura bitters','old fashioned bitters','mole bitters','fennel bitters','apple bitters','black walnut bitters',
                                  'rhubarb bitters','bittermens burlesque bitters','aromatic bitters','peychaud bitters','bitter lemon','bob\'s bitters abbott\'s bitter','gentian bitters','hellfire bitters',
                                  'the bitter truth jerry thomas\' own decanter bitter', 'cardamom bitters', 'cognac', 'vs cognac','salt tincture','hine rare vsop','amer picon','gran classico','almond extract','almond flavoring',
                                  'christmas spirit','pineau des charentes','sarsaparilla','pastis','armagnac','cachaça','cachaça (leblon)','fernet branca','elderflower cordial']
ingredients_transformation['rum_ingr'] = ['rum', 'coconut rum', 'pineapple rum','light rum','english harbour rum', 'pink pigeon rum', 'aged rum', 'white rum', 'black rum', 'banks 5 rum', 'añejo rum', 'spiced rum',
                               'dark rum', 'malibu rum', 'mr. boston rum', 'gold rum', 'bacardi rum', 'jamaica rum', '5-proof rum', 'anejo rum', '151 rum','smith & cross','bacardi limon','arrack']
ingredients_transformation['water_ingr'] = ['water', 'coconut water', 'tonic water','sparkling water','boiling water','cold water','carbonated water','topo chico mineral water','seltzer water','rose water','hot water',
                                 'water (divided)','water melon','water (distilled)','of water','quarts water','ice','crushed ice','to 4 ice','of ice']
ingredients_transformation['milk_ingr'] = ['milk', 'coconut milk','condense milk','almond milk','chocolate milk','whole milk','milk (or cream)','soy milk','cups  milk']
ingredients_transformation['tequila_ingr'] = ['tequila','blanco tequila', 'reposado tequila', 'gold tequila', 'anejo tequila', '. tequila', 'tequila rose', 'silver tequila', 'of tequila' , 'prickly pear tequila']
ingredients_transformation['olive_ingr'] = ['olive','olive juice', 'olive brine', 'green olive']
ingredients_transformation['juice_ingr'] = ['juice','grape juice', 'lychee juice','yuzu juice','pomegranate juice','tomato juice','strawberry juice','rhubarb juice','cranberry juice','lemon juice','mango juice','apple juice','fruit juice',
                                 'clam juice','raspberry juice','blueberry juice', 'carrot juice', 'pomegranite juice', 'agave nectar', 'peach nectar', 'mango nectar', 'apricot nectar', 'tamarind nectar']
ingredients_transformation['coffee_ingr'] = ['coffee', 'cold-brew coffee', 'espresso coffee', 'hot coffee', 'cold brewed coffee', 'coffee beans', 'tables instant coffee','espresso']
ingredients_transformation['lemon_ingr'] = ['lemon','fresh lemon', 'lemon sour', 'lemon sorbet', 'lemon peel','lemon slice','lemon twist','lemon wheel','lemon zest','lemon (cut into )','lemon (juied)','lemon, thinly slied','emon',
                                 'lime sour','fresh lime','lime juice','lime','lime vodka', 'lime peel','lime twist','lime wheel', 'lime wedge', 'lime cordial', 'lime (cut into )', 'lime zest', 'lime (slied)',
                                 'lime slice','lime(cut into )', 'lime (juied)', 'lemon wedge','lemons (slied)','ime','squeeze of ime']
ingredients_transformation['egg_ingr'] = ['egg','egg white', 'egg yolk', 'whole egg', 'egg yok', 'sma egg', 'arge egg']
ingredients_transformation['tea_ingr'] = ['tea', 'chai tea','iced tea','tea (chilled)','chamomile tea','black tea','green tea leaves','hibiscus tea']
ingredients_transformation['cider_ingr'] = ['cider','apple cider', 'hard cider', 'sparkling cider', 'fresh cider']
ingredients_transformation['grapefruit_ingr'] = ['grapefruit', 'grapefruit twist','grapefruit wedge','grapefruit slice','grapefruit peel']
ingredients_transformation['ginger_ingr'] = ['ginger', 'fresh ginger','crystallised ginger','ground ginger','thin slice ginger']
ingredients_transformation['butter_ingr'] = ['butter','spiced butter','table butter','butter (softened)']
ingredients_transformation['cream_ingr'] = ['cream','heavy cream','whipped cream','whipping cream','light cream','irish cream','chocolate ice-cream','sweet cream','amarula cream','creme de banane','coconut cream',
                                 'fresh sweet n\' sour','sweet and sour','vanilla cream','creme de cassis','creme de mure','creme de banana','creme de noyaux','creme de violette','creme de menthe','creme yvette']
ingredients_transformation['sauce_ingr'] = ['sauce','worcestershire sauce','tabasco sauce','soy sauce','hot sauce','tomato catsup']
ingredients_transformation['cloves_ingr'] = ['cloves', 'whole cloves','ground cloves']
ingredients_transformation['cucumber_ingr'] = ['cucumber','cucumber peel','cucumber juice','cucumber slice']

# garnish
ingredients_transformation['common_ingr'] = ['citric acid', 'pineapple gum', 'pineapple', 'pineapple wedge', 'pineapple leaves', 'cup d pineapple',
                                        'green chartreuse', 'yellow chartreuse', 'tempus fugit kina l\'aero d\'or', 'tonic', 'fever-tree tonic', 'drambuie', 'horchata', 'briottet creme de framboise (raspberry)','crème yvette',
                                        'grand marnier', 'creme de cacao', 'crème de cacao', 'salers gentiane aperitif', 'nutmeg', 'ground nutmeg', 'grated nutmeg','montelobos mezcal','mezcal',
                                        'salt','celery salt','coarse salt','table salt', 'cranberries', 'package cranberries','cup cranberries','cocchi americano', 'creole shrub', 'blueberry shrub',
                                        'pumpkin puree', 'peach puree','clove tincture','clove','vanilla','vanilla extract','s vanilla','citrus agave blend','sugar', 's sugar','molasses','pomegranate molasses','h by hine',
                                        'watermelon','agave','honey','table honey','pear','ripe pear','pear slice','strawberry','strawberry schnapps','whole strawberry','strawberries','strawberries (sliced)','cocoa powder',
                                        'licorice root','mango','mango slice','d mango','angelica root','surge','mint','allspice','allspice berries','ground allspice','cherry','cocktail cherry','apple','apple slice','anis',
                                        'cardamom','cherries','forbidden fruit','fruit','passion fruit','hot chocolate','chocolate','dark chocolate','berries','anise','star anise','banana','ripe banana','yoghurt',
                                        'food coloring','coriander','coriander seeds','prepared horseradish','vinegar','balsamic vinegar','peach slice','red grapes','white grapes','cold beef bouillon','blueberries',
                                        'frozen blueberries','peach','peach purée','b & b','thyme sprig','sprigs avender','vania bean','basil leaves','cup raspberries','raspberries','edible flower','blackberries',
                                       'cocktail onion','ardamom pods','mustard','red appe','bay leaf','bay eaves','smoked paprika','our mix','3- package gelatin (any flavor)','s gelatin','sage eaves','fennel seed',
                                       'dried rose buds','marmalade','blueberry preserve','coco lopez','half and half','raspberry preserve','pea-sized doop of wasabi paste','chambord','ilantro leaves','oves','eggnog','sliesuumber']

def find_group_for_ingredient(ingr_name):
    for ind, key in enumerate(ingredients_transformation.keys()):
        if ingr_name in ingredients_transformation.get(key):
            return key, ind
    raise Exception('Wrong ingredient name')

def prepare_data(param, scaler):
    data = [0 for x in range(len(ingredients_transformation))]
    for item in param:
        ingr_group, index = find_group_for_ingredient(item[0])
        value = measures[findMeasure(item[2], measures)] * item[1]
        data[index] = value
    return scaler.transform(np.asarray(data).reshape(1,-1))