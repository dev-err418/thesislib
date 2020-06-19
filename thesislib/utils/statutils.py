"""
This module contains helper functions for computing relevant statistics for the dataset
"""


def compute_A_within(class_key, categories, dataframe):
    """
    Computes the A-between index for different classes in a categorical dataset.
    See http://www.urbanlab.org/articles/Lieberson_1969_MeasuringPopulationDiversity.pdf
    Specifically the sectional on A-indices for attitudinal data
    :param class_key: a string which is the key by which the different classes are identified
    :param categories: a list of the the column keys which correspond to the categorical data
    :param dataframe: the data frame
    :return: a dictionary of class: A-within key value pairs
    """

    # get the list of classes
    classes = dataframe[class_key].unique().tolist()

    # initialize the within A index
    A_within = {itm: 0 for itm in classes}

    # group by the classes
    grp = dataframe.groupby([class_key])

    # iterate over the groups
    for itm, df in grp.__iter__():
        df = df[categories]
        # compute the probabilities per category; assumption is that each category is either a 0 or 1
        _tmp = (df.sum() / df.shape[0])

        # compute the summation across all categories
        _tmp = _tmp.transform(lambda x: x - x*x).sum()

        # finally compute the A-index
        _tmp = _tmp * 2 / len(categories)

        A_within[itm] = _tmp

    return A_within


def compute_A_between(class_key, categories, dataframe):
    """
    Computes the A-between index for different classes in a categorical dataset.
    See http://www.urbanlab.org/articles/Lieberson_1969_MeasuringPopulationDiversity.pdf
    Specifically the sectional on A-indices for attitudinal data
    :param class_key: a string which is the key by which the different classes are identified
    :param categories: a list of the the column keys which correspond to the categorical data
    :param dataframe: the data frame
    :return: a nested dictionary representing the A-between matrix for the given classes
    """
    classes = dataframe[class_key].unique().tolist()

    # initialize the between A index
    A_between = {itm: {itm_1: 0 for itm_1 in classes} for itm in classes}

    num_categories = len(categories)

    prob_collection = {}
    # for all the conditions, compute the probabilities
    grp = dataframe.groupby([class_key])
    for itm, df in grp.__iter__():
        df = df[categories]
        prob_collection[itm] = df.sum()/df.shape[0]

    for idx in range(len(classes)):
        itm_0 = classes[idx]
        pr_0 = prob_collection[itm_0]
        for jdx in range(idx +1, len(classes)):
            itm_1 = classes[jdx]
            pr_1 = prob_collection[itm_1]

            _tmp = ((pr_0 + pr_1).sum() - 2 * (pr_0 * pr_1).sum())/num_categories
            A_between[itm_0][itm_1] = _tmp
            A_between[itm_1][itm_0] = _tmp

    return A_between
