from logic.data_generator import read_from_pickle


if __name__ == '__main__':
    map_path = '../resources/data/tlv_data/100_100_20_3_33/tlv_data_100_100_20_3_33.pkl'
    map_object = read_from_pickle(map_path)
    print(map_object.get_map_details())