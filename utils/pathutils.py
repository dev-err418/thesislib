import os


def get_data_directory():
    """
    Returns the path to the data directory
    :return: string
    """
    data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    return data_directory


def get_data_file(relative_path):
    """
    This function expects that there is a `data` folder just outside the top level parent package.
    It returns the absolute path of the data directory concatenated with the provided relative path
    :param relative_path: The relative path to be fetched from the data directory
    :return:
    """
    return os.path.join(get_data_directory(), relative_path)
