import os
import sys
import json

# directory structure: {
#   .
#   -WFLW_images/Prepr_dlib_format/....jpg
#                                 /....jpg.landmarks.json
#   -prepr_xml/
# }


def emit_for_face(face):
    rect = face["bb"]
    landmarks = face["landmark"]
    landmark_xmls = []

    for i in range(len(landmarks)):
        if i < 10:
            index = "0" + str(i)
        else:
            index = str(i)
        x = int(round(landmarks[str(i)]["x"]))
        y = int(round(landmarks[str(i)]["y"]))
        landmark_xmls.append('<part name="{0}" x="{1}" y="{2}" />'.format(index, x, y))
    return '<box top="{0}" left="{1}" width="{2}" height="{3}">\n{4}\n</box>'.format(
        rect["top"], rect["left"], rect["width"], rect["height"], '\n'.join(landmark_xmls)
    )

if __name__ == "__main__":
    try:
        json_path = sys.argv[1]
        xml_path = sys.argv[2]
    except:
        print("error: please write image directory path and xml save directory")

    json_file = open(json_path, 'r')
    faces_json = json.load(json_file)

    image_xmls = ''
    for image_path, face in faces_json.items():
        face_xmls = emit_for_face(face)
        image_xmls += '<image file="{0}">\n{1}\n</image>'.format(image_path, face_xmls)
        image_xmls += '\n'

    root_xml = '<dataset>\n<images>\n{0}</images>\n</dataset>'.format(image_xmls)

    xml_file = open(xml_path, 'w')
    xml_file.write(root_xml)
    xml_file.close()
