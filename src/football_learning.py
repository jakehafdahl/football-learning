import data_parser
import numpy as np
from polynomial_season import PolynomialSeason
from neuralnet_season import NeuralNetworkSeason

def build_data_for_position(players, position):
    """
        Builds training data for a given position
        """
    position_players = [player for player in players if player.position() == position and (len(player.seasons()) > 4)]
    print("There are %s players for position %s" % (len(position_players), position))
    
    X = []
    y = []
    for player in position_players:
        seasons = player.seasons()
        for i in range(0, len(seasons) - 4):
            set = np.append([],(seasons[i].format_data(),
                                seasons[i + 1].format_data(),
                                seasons[i + 2].format_data()))
            X.append(set)
            y.append(seasons[i + 3].format_data())


    print("There are %s training examples for position %s, with %s features" % (len(X),position,len(X[0])))
    
    return { 'train': X, 'target': y}


def format_player_data(players):
    """
        Gets player data formatted for machine learning
        Returns:
        A dictionary with 2 keys:
        'train' - set of data to train on
        'target' - the target data for the training set
    """
    X_qb = build_data_for_position(players,"qb")
    X_rb = build_data_for_position(players,"rb")
    X_wr = build_data_for_position(players,"wr")
    X_te = build_data_for_position(players,"te")
    
    return { 'WR': X_wr, 'QB': X_qb, 'RB': X_rb, 'TE': X_te }

def partition_data(data, target):
    sets = len(data)
    eighty = sets * 8 / 10
    X = np.matrix(data)
    y = np.matrix(target)
    print("eighty is %s" % eighty)
    return X[:eighty,:], y[:eighty,:], X[eighty:,:], y[eighty:,:]


def evaluate_model(X_test, y_test, model):
    error = []
    for i in range(0,len(X_test)):
        diff = np.subtract(model.predict(X_test[i]),y_test[i])
        error.append(diff)

    avg = np.average(error,0)
    return avg


def run():
    players = data_parser.get_master_player_list()
    seasons = data_parser.get_player_seasons_list()
    games = data_parser.get_player_games_list()
    print("%s games parsed" % len(games))
    built = data_parser.combine_players_seasons_games_lists(players, seasons, games)
    players_list = [value for (key, value) in built.iteritems()]
    data = format_player_data(players_list)
    #model = PolynomialSeason()
    rbnet = NeuralNetworkSeason(position="RB")
    wrnet = NeuralNetworkSeason(position="WR")
    qbnet = NeuralNetworkSeason(position="QB")
    tenet = NeuralNetworkSeason(position="TE")

    rbtrain_data, rbtrain_target, rbtest_data, rbtest_target = partition_data(data['RB']['train'], data['RB']['target'])
    wrtrain_data, wrtrain_target, wrtest_data, wrtest_target = partition_data(data['WR']['train'], data['WR']['target'])
    qbrbtrain_data, qbtrain_target, qbtest_data, qbtest_target = partition_data(data['QB']['train'], data['QB']['target'])
    terbtrain_data, tetrain_target, tetest_data, tetest_target = partition_data(data['TE']['train'], data['TE']['target'])
    
    #model.train(train_data, train_target)
    rbnet.train(rbtrain_data, rbtrain_target)
    wrnet.train(wrtrain_data, wrtrain_target)
    qbnet.train(qbtrain_data, qbtrain_target)
    tenet.train(tetrain_data, tetrain_target)

    #avg = evaluate_model(test_data, test_target, model)

    rbnetavg = evaluate_model(rbtest_data, rbtest_target, rbnet)
    wrnetavg = evaluate_model(wrtest_data, wrtest_target, wrnet)
    qbnetavg = evaluate_model(qbtest_data, qbtest_target, qbnet)
    tenetavg = evaluate_model(tetest_data, tetest_target, tenet)
    
    #print("average errors for linear reg are %s" % avg)
    print("average errors for neural net RB reg are %s" % rbnetavg)
    print("average errors for neural net WR reg are %s" % wrnetavg)
    print("average errors for neural net QB reg are %s" % qbnetavg)
    print("average errors for neural net TE reg are %s" % tenetavg)


if __name__ == '__main__':
    run()