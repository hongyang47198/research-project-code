import ijson
import json

file_name = '96_train.json'

with open(file_name,'r') as f:

    parser = ijson.parse(f)

    object_id = 0

    parser_object = {}
    coordinates = []
    target_object = []
    text = []

    n = 0

    for prefix, event, value in parser:
        print prefix, event, value
        if str(prefix) == "rows.item.doc.text":
            text.append(value)
        if str(prefix) == "rows.item.doc.coordinates":
            coordinates.append(value)
        if str(prefix) == "rows.item.id":
            if len(target_object) >1 and target_object[1] != None:
                parser_object[object_id] = target_object
    #                 print target_object
                target_object = []
                object_id +=1
                print object_id
            else:
                target_object = []
        if n == 1000:
            break
        n += 1


    f.close()


