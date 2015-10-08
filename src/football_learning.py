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
    net = NeuralNetworkSeason(position="RB")

    train_data, train_target, test_data, test_target = partition_data(data['RB']['train'], data['RB']['target'])
    
    #model.train(train_data, train_target)
    net.train(train_data, train_target)

    #avg = evaluate_model(test_data, test_target, model)

    netavg = evaluate_model(test_data, test_target, net)
    
    #print("average errors for linear reg are %s" % avg)
    print("average errors for neural net reg are %s" % netavg)


if __name__ == '__main__':
    run()