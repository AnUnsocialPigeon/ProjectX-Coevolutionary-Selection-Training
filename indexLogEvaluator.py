fileDir = "./indexeslog.text"

used = {}
lineCount = 0
l1 = False
l2 = False

with open(fileDir, "r") as file:
    # Skip the first line
    for line in file:
        lineCount += 1
        if lineCount == 1:
            for i in line.split(', '):
                if int(i) > 1:
                    l1 = True
                    print(f"Line 1 had: {i}")
        if lineCount == 2:
            for i in line.split(', '):
                if int(i) > 1:
                    l2 = True
                    print(f"Line 2 had: {i}")
        for i in line.split(', '):
            index = int(i)
            if index in used.keys():
                used[index]+=1
                continue
            used[index] = 1

sortedValues = sorted(used.items(), key=lambda x: x[1])

keys = used.keys()
values = used.values()

string = ""
maxKey = -1
maxKeyVal = -1

for key, value in sortedValues:
    if key > maxKey:
        maxKey = key
        maxKeyVal = value
        print(f"New max: {maxKey},{maxKeyVal}")
    string += f"{key}:\t{value}\n"

input("Press enter for results:")

print(string)

print(f"Max index given: {max(keys)} ({used[max(keys)]})")
print(f"Indecies in training: {60000}")
print(f"Line 1 was not binary: {l1}")
print(f"Line 2 was not binary: {l2}")
print(f"Unique indexes given: {len(keys)}")
print(f"Unique indexes not given: {50000 - len(keys)}")
print(f"Prey data subsets given: {lineCount}")

