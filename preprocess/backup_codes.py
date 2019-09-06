def check_yago_types_of_at_least_10():
    yago_file = 'yago_types_of_at_least_10.json'
    yago = json.load(fp = open(join(data_dir, res_dir, yago_file), 'r'))
    print(yago['Height105137165'])


def count_wiki_yago_types():
    yago_file = 'wiki-title-yago-types.txt'
    with open(join(data_dir, res_dir, yago_file), 'r') as r:
        r.readline()
        line = r.readline()
        type_set = set()
        i = 0
        while line:
            items = line.split(',')

            types = items[2].split(';')
            type_set.update(types)

            if i % 100000 == 0:
                print(i, int(items[0]), items[1], types)
            i += 1
            line = r.readline()

    type_set = list(type_set)
    type_set.sort()
    with open(join(data_dir, 'type_set.txt'), 'w') as w:
        for t in type_set:
            w.write(t + '\n')