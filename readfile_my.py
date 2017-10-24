import ijson
import json

file_name = 'db95.json'

with open(file_name,'r') as f:

  parser = ijson.parse(f)

  n = 0

  object_id = 0

  parser_object = {}
  tweets_text = []
  tweets_coordinates = []
  coordinates = []
  name = []
  creat_time = []


  for prefix, event, value in parser:
    if str(prefix) == "rows.item.doc.text" and str(event) == "string":
      try:
        s = value.encode('utf-8')
      except:
        s = value
      # print s
      tweets_text.append(s)
    if str(prefix) == "rows.item.doc.coordinates.coordinates.item" and str(event) == "number":
        coordinates.append(str(value))
    if str(prefix) == "rows.item.doc.user.screen_name" and str(event) == "string":
      name.append(str(value))
    if str(prefix) == "rows.item.doc.created_at" and str(event) == "string":
      creat_time.append(str(value))
    if str(prefix) == "rows.item.doc.place.bounding_box" and str(event) == "end_map":
      if coordinates:
        tweets_coordinates.append(coordinates)
      else:
        tweets_coordinates.append([0,0])
      coordinates = []
      object_id += 1
      print object_id

print "start writing"
print len(tweets_coordinates)
print len(creat_time)
print len(tweets_text)
print len(name)

with open('new/95_result_text.txt','w') as wf:
  n = 0
  for text in tweets_text:
    st = str(n) + " " + str(text).replace("\n","")+"\n"
    n +=1
    wf.write(st)
  wf.close()

with open('new/95_result_coordinate.txt','w') as cf:
  n =0
  for coordinate in tweets_coordinates:
    co = str(n) + " "+ str(coordinate) + "\n"
    n +=1
    cf.write(co)
  cf.close()

with open('new/name.txt','w') as f:
  name = set(name)
  for names in name:
    nm = str(names) + "\n"
    f.write(nm)
  f.close()

with open('new/95_result_time.txt','w') as f:
  n = 0
  for time in creat_time:
    ct = str(n) + " " + str(time) + "\n"
    n += 1
    f.write(ct)
  f.close()

