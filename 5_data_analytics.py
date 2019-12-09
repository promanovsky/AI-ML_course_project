import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from common.tools import forest_regression_test, forest_classification_test, gradient_boosting_classification_test, \
    gradient_boosting_regression_test, showXGBTrainImportance, tree_classification_test
from decompositions import doPca_decomposition_demonstration, do_tsne_decomposition_demonstration, do_lda_decomposition_demonstration, \
    do_lle_decomposition_demonstration,do_AE_decomposition_demonstration
"""
Этап 5
Данный скрипт реализует функцию-бенчмарк которая для указанного датасета запускает цикл создания моделей регрессии и 
классификации, снимает статистику и рисует графики/диаграммы
"""

def do_experiment(dataset_path):
    curr_dir = os.path.abspath(os.curdir)
    df = pd.read_csv(curr_dir + dataset_path)
    print(df.shape)

    columns_to_scale = list(df.columns)
    columns_to_scale.remove('Cocktail Name')
    columns_to_scale.remove('rating')
    #print(len(columns_to_scale), columns_to_scale)

    #min_max_scaled = minmax_scale(df[columns_to_scale])
    #min_max_scaled_columns = pd.DataFrame(min_max_scaled, columns=columns_to_scale)
    #table = pd.DataFrame(pd.concat((df['Cocktail Name'], min_max_scaled_columns, df['rating']), axis = 1))
    #table.to_csv(curr_dir +'/datasets/processed_minmax_scaler_dataset.csv', index=False)

    scaler = StandardScaler()
    scaler.fit(df[columns_to_scale])
    standart_scaled = scaler.fit_transform(df[columns_to_scale])
    standart_scaled_columns = pd.DataFrame(standart_scaled, columns=columns_to_scale)
    #table = pd.DataFrame(pd.concat((df['Cocktail Name'], standart_scaled_columns, df['rating']), axis = 1))
    #table.to_csv(curr_dir +'/datasets/processed_standart_scaler_dataset.csv', index=False)

    showXGBTrainImportance(standart_scaled_columns, df['rating'], columns_to_scale)
    #showXGBTrainImportance(min_max_scaled_columns, df['rating'], columns_to_scale)

    Y = df['rating']
    # классификация по рейтингу
    tree_classification_test(standart_scaled, Y)
    forest_classification_test(standart_scaled, Y)
    gradient_boosting_classification_test(standart_scaled, Y)

    # регрессия (предсказание рейтинга)
    #forest_regression_test(standart_scaled, Y)
    #gradient_boosting_regression_test(standart_scaled, Y)

    return standart_scaled, df['rating']


# построение моделей на датасете до группировки (порядка 800 столбцов)
scaled_data, label_df = do_experiment('/datasets/reducing_dataset.csv')
# doPca_decomposition_demonstration(scaled_data, label_df)
# do_tsne_decomposition_demonstration(scaled_data, label_df)
# do_lda_decomposition_demonstration(scaled_data, label_df,3)
# do_lda_decomposition_demonstration(scaled_data, label_df,30)
# do_lle_decomposition_demonstration(scaled_data, label_df,3)
# do_lle_decomposition_demonstration(scaled_data, label_df,30)
# do_AE_decomposition_demonstration(scaled_data, label_df)

# построение моделей на датасете после группировки но с нулями вместо пустых данных
scaled_data, label_df = do_experiment('/datasets/grouped_columns.csv')
# doPca_decomposition_demonstration(scaled_data, label_df)
# do_tsne_decomposition_demonstration(scaled_data, label_df)
# do_lda_decomposition_demonstration(scaled_data, label_df,3)
# do_lda_decomposition_demonstration(scaled_data, label_df,30)
# do_lle_decomposition_demonstration(scaled_data, label_df,3)
# do_lle_decomposition_demonstration(scaled_data, label_df,30)
# do_AE_decomposition_demonstration(scaled_data, label_df)

# построение моделей на датасете с заполненными метками на основании регрессии
scaled_data, label_df = do_experiment('/datasets/grouped_columns_filled_with_regression.csv')
# doPca_decomposition_demonstration(scaled_data, label_df)
# do_tsne_decomposition_demonstration(scaled_data, label_df)
# do_lda_decomposition_demonstration(scaled_data, label_df,3)
# do_lda_decomposition_demonstration(scaled_data, label_df,30)
# do_lle_decomposition_demonstration(scaled_data, label_df,3)
# do_lle_decomposition_demonstration(scaled_data, label_df,30)
# do_AE_decomposition_demonstration(scaled_data, label_df)

# построение моделей на датасете с заполненными метками на основании кластеризации
scaled_data, label_df = do_experiment('/datasets/grouped_columns_filled_with_clusterization.csv')
# doPca_decomposition_demonstration(scaled_data, label_df)
# do_tsne_decomposition_demonstration(scaled_data, label_df)
# do_lda_decomposition_demonstration(scaled_data, label_df,3)
# do_lda_decomposition_demonstration(scaled_data, label_df,30)
# do_lle_decomposition_demonstration(scaled_data, label_df,3)
# do_lle_decomposition_demonstration(scaled_data, label_df,30)
# do_AE_decomposition_demonstration(scaled_data, label_df)