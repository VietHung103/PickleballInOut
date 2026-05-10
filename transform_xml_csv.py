import xml.etree.ElementTree as ET
import csv

xml_path = r"C:\AI\pickleball\data\game_5\clip_1\frames\annotations.xml"
output_csv = r"C:\AI\pickleball\data\game_5\clip_1\frames\tracknet_label_clip_1.csv"

tree = ET.parse(xml_path)
root = tree.getroot()

data = []

for image in root.findall("image"):
    frame_id = int(image.get("id"))
    width = int(image.get("width"))
    height = int(image.get("height"))
    
    points = image.find("points")
    
    if points is not None:
        xy = points.get("points")
        x, y = map(float, xy.split(","))
        visibility = 1
    else:
        x, y = -1, -1
        visibility = 0

    data.append([frame_id, x, y, visibility])

# Save CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "x", "y", "visibility"])
    writer.writerows(data)

print("Conversion complete!")