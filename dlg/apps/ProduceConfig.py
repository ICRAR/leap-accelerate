import csv
import json

# should be read from DALiuGE
NUMBER_OF_COPIES = 1
NUM_STATIONS = 126
DIRECTIONS_FILENAME = "directions.csv"
MEASUREMENT_SET_FILENAME = "/Users/james/working/leap-accelerate/testdata/1197638568-32.ms"
IMPLEMENTATION = 'eigen'


def readDirections():
    directions = []

    with open(DIRECTIONS_FILENAME) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            x = float(row[0])
            y = float(row[1])
            directions.append([x,y])

    return directions


def createConfig(numStations, directions, implementation):
    return {
        'numStations': numStations,
        'filePath': MEASUREMENT_SET_FILENAME,
        'directions': directions,
        'implementation': implementation
    }


def main():
    directions = readDirections()

    for i in range(NUMBER_OF_COPIES):
        config = createConfig(NUM_STATIONS, directions, IMPLEMENTATION)
        print(json.dumps(config))


if __name__ == "__main__":
    main()
