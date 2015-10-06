from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import data_parser
import numpy as np


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


def train_linear_regression(train,target):
    """
        Trains via linear regression from the target and train data
        Arguments:
        train : an array of training data
        target: an array of associated target data
        Returns:
        The trained model
        """
    model = Pipeline([('poly',PolynomialFeatures(degree=2)),
                      ('linear', LinearRegression(fit_intercept=False, normalize=True, copy_X=True))])
    model = model.fit(train, target)
    return model


def run():
    players = data_parser.get_master_player_list()
    seasons = data_parser.get_player_seasons_list()
    games = data_parser.get_player_games_list()
    print("%s games parsed" % len(games))
    built = data_parser.combine_players_seasons_games_lists(players, seasons, games)
    players_list = [value for (key, value) in built.iteritems()]
    data = format_player_data(players_list)
    model = train_linear_regression(data['WR']['train'], data['WR']['target'])
    print("training on %s training sets" % len(data['WR']['train']))
    print("predicted %s " % model.predict( data['WR']['train'][-2]))
    print("actual %s " % data['WR']['target'][-2])

if __name__ == '__main__':
    run()