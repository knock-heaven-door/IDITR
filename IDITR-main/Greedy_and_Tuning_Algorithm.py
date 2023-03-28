import csv
import pandas as pd


# Determine if there is a path from x_m to x_n in the graph
def find_path(x_m, x_n):
    if depth[x_m] >= depth[x_n]:
        return False
    elif depth[x_m] < depth[x_n]:
        queen_m = [x_m]
        a_m = x_m
        find = [-1] * 607
        find[x_m] = 1
        while len(queen_m) != 0:
            a_m = queen_m.pop(0)
            if a_m == x_n:
                return True
            for item_m in Graph[a_m]:
                if find[item_m] < 0:
                    find[item_m] = 1
                else:
                    continue
                queen_m.append(item_m)
        return False


# Determine if x_m and x_n are reachable in the graph
def find_path_2(x_m, x_n):
    if depth[x_m] == depth[x_n]:
        return False
    elif depth[x_m] < depth[x_n]:
        queen_m = [x_m]
        a_m = x_m
        find = [-1] * 607
        find[x_m] = 1
        while len(queen_m) != 0:
            a_m = queen_m.pop(0)
            if a_m == x_n:
                return True
            for item_m in Graph[a_m]:
                if find[item_m] < 0:
                    find[item_m] = 1
                else:
                    continue
                queen_m.append(item_m)
        return False
    else:
        queen_m = [x_n]
        a_m = x_n
        find = [-1] * 607
        find[x_n] = 1
        while len(queen_m) != 0:
            a_m = queen_m.pop(0)
            if a_m == x_m:
                return True
            for item_m in Graph[a_m]:
                if find[item_m] < 0:
                    find[item_m] = 1
                else:
                    continue
                queen_m.append(item_m)
        return False


start = list(range(0, 607))
Graph = []

RAW = [[0]*607 for _ in range(607)]
WAW = [[0]*607 for _ in range(607)]
WAR = [[0]*607 for _ in range(607)]
correlation = [[0]*607 for _ in range(607)]

# print(start)

# Read in the csv file and determine the root node of the graph
with open('control dependency.csv') as f:
    graph = csv.reader(f)
    for line in graph:
        line_1 = []
        # print(line)
        for item in line[1:]:
            line_1.append(int(item))
            if int(item) in start:
                start.remove(int(item))
        Graph.append(line_1)
f.close()

# print(start)

depth = [-1] * 607

for item in start:
    depth[item] = 0

# Start a breadth search from start to determine the node depth
queen = start.copy()
d = 0
while len(queen) != 0:
    a = queen.pop(0)
    for item in Graph[a]:
        if depth[item] < depth[a] + 1:
            depth[item] = depth[a] + 1
        else:
            continue
        queen.append(item)
# print(depth)

depth.append(1000)


# Find data dependency
data_correlation = []
with open('data dependency.csv') as f:
    data = csv.reader(f)
    for line in data:
        data_correlation.append(line)
f.close()

# print(data_correlation)

# Build W, R dictionaries
W_dict = {}
for x_num in range(0, 824):
    dict_line = []
    x_char = 'X' + str(x_num)
    for i in range(0, 1214, 2):
        if x_char in data_correlation[i]:
            dict_line.append(int(i/2))
    W_dict[x_char] = dict_line
# print(W_dict)

R_dict = {}
for x_num in range(0, 824):
    dict_line = []
    x_char = 'X' + str(x_num)
    for i in range(1, 1214, 2):
        if x_char in data_correlation[i]:
            dict_line.append(int(i/2))
    R_dict[x_char] = dict_line
# print(R_dict)

# Find out WAW dependency
for x_num in range(0, 824):
    x_char = 'X' + str(x_num)
    for i in W_dict[x_char]:
        for j in W_dict[x_char]:
            if find_path(i, j):
                WAW[i][j] = 1
                # print(str(i) + ',' + str(j))
# print(WAW[25][91])

# Find out WAR dependency
for x_num in range(0, 824):
    x_char = 'X' + str(x_num)
    for i in R_dict[x_char]:
        for j in W_dict[x_char]:
            if find_path(i, j):
                WAR[i][j] = 1
                # print(str(i) + ',' + str(j))

# Find out RAW dependency
for x_num in range(0, 824):
    x_char = 'X' + str(x_num)
    for i in W_dict[x_char]:
        for j in R_dict[x_char]:
            if find_path(i, j):
                RAW[i][j] = 1
                # print(str(i) + ',' + str(j))

# df = pd.DataFrame(WAW)
# df.to_csv('WAW.csv')
# df = pd.DataFrame(WAR)
# df.to_csv('WAR.csv')
# df = pd.DataFrame(RAW)
# df.to_csv('RAW.csv')

control_relation = [[0]*607 for _ in range(607)]
# Find out control dependency
neighbor_child = [-1]*608

for i in range(0, 607):
    generation = Graph[i].copy()
    if len(generation) == 0:
        generation = [607]
    while len(generation) != 1:
        generation.sort(key=lambda x: depth[x])
        child = generation.pop(0)
        control_relation[i][child] = 1
        for grand_child in Graph[child]:
            if grand_child in generation:
                continue
            else:
                generation.append(grand_child)
    neighbor_child[i] = generation[0]
    if generation[0] <= 606:
        control_relation[i][generation[0]] = 0


node = list(range(0, 608))
node.sort(reverse=True, key=lambda x: depth[x])

for i in node:
    if neighbor_child[i] < 0:
        continue
    n_child = neighbor_child[i]
    if n_child > 606:
        continue
    for j in range(0, 607):
        if control_relation[n_child][j] == 1:
            control_relation[i][j] = 1

# df = pd.DataFrame(control_relation)
# df.to_csv('control_relation.csv')

for i in range(607):
    for j in range(607):
        if WAW[i][j] == 1:
            correlation[i][j] = 2
        if WAR[i][j] == 1:
            correlation[i][j] = 1
        if RAW[i][j] == 1:
            correlation[i][j] = 2
        if control_relation[i][j] == 1:
            correlation[i][j] = 1

# df = pd.DataFrame(correlation)
# df.to_csv('correlation.csv')

test = []
for i in range(607):
    for j in range(i, 607 - i):
        if correlation[i][j] * correlation[j][i] != 0:
            print("error appeared in (" + str(i) + "," + str(j) + ")")

node = list(range(0, 607))
node.sort(reverse=False, key=lambda x: depth[x])
# print(node)
handled_node = []
path_depth = [-1] * 607
min_depth = 0

# Give the topological ordering of the nodes based on the final obtained path constraint matrix
while len(node) != 0:
    temp_node = node.pop(0)
    min_depth = 0
    for i in handled_node:
        if correlation[i][temp_node] == 1:
            if min_depth < path_depth[i] + 0.001:
                min_depth = path_depth[i] + 0.001
        elif correlation[i][temp_node] == 2:
            if min_depth < int(path_depth[i]) + 1:
                min_depth = int(path_depth[i]) + 1
    path_depth[temp_node] = min_depth
    handled_node.append(temp_node)

# for i in range(0, 607):
#     if path_depth[i] < 1:
#         print(str(i) + ',' + str(path_depth[i]))

# A scheduling scheme for the original problem is given by the greedy algorithm,
# specifically adding as many nodes as possible to the interior of each level of the flow in topological order
node = list(range(0, 607))
node.sort(reverse=False, key=lambda x: path_depth[x])
# print(node)

data_resources = []
resources = [[0,0,0,0] for _ in range(607)]

with open('resource requirement.csv') as f:
    data = csv.reader(f)
    for line in data:
        data_resources.append(line[1:])
f.close()

data_resources.pop(0)

for i in range(0, 607):
    for j in range(0, 4):
        resources[i][j] = int(data_resources[i][j])

# print(resources)

R_TCAM = [1] * 607
R_HASH = [2] * 607
R_ALU = [56] * 607
R_QUALIFY = [64] * 607
FOLD_TCAM = [1] * 16
FOLD_HASH = [3] * 16
even_TCAM_count = 0

pipeline = [[] for _ in range(607)]
degree = 0
last_node = node[0]

# Start putting nodes into the pipeline
while len(node) != 0:
    temp_node = node.pop(0)
    if int(path_depth[temp_node]) - int(path_depth[last_node]) >= 1:
        degree += 1
    if (R_TCAM[degree] >= resources[temp_node][0]) & (R_HASH[degree] >= resources[temp_node][1])\
        & (R_ALU[degree] >= resources[temp_node][2]) & (R_QUALIFY[degree] >= resources[temp_node][3]):
        R_TCAM[degree] -= resources[temp_node][0]
        R_HASH[degree] -= resources[temp_node][1]
        R_ALU[degree] -= resources[temp_node][2]
        R_QUALIFY[degree] -= resources[temp_node][3]
    else:
        node.insert(0, temp_node)
        degree += 1
        continue
    if degree <= 31:
        if (FOLD_TCAM[degree % 16] >= resources[temp_node][0]) & \
                (FOLD_HASH[degree % 16] >= resources[temp_node][1]):
            FOLD_TCAM[degree % 16] -= resources[temp_node][0]
            FOLD_HASH[degree % 16] -= resources[temp_node][1]
        else:
            R_TCAM[degree] += resources[temp_node][0]
            R_HASH[degree] += resources[temp_node][1]
            R_ALU[degree] += resources[temp_node][2]
            R_QUALIFY[degree] += resources[temp_node][3]
            node.insert(0, temp_node)
            degree += 1
            continue
    if (degree % 2 == 0) & (resources[temp_node][0] == 1):
        if even_TCAM_count < 5:
            even_TCAM_count += 1
        else:
            R_TCAM[degree] += resources[temp_node][0]
            R_HASH[degree] += resources[temp_node][1]
            R_ALU[degree] += resources[temp_node][2]
            R_QUALIFY[degree] += resources[temp_node][3]
            if degree <= 31:
                FOLD_TCAM[degree % 16] += resources[temp_node][0]
                FOLD_HASH[degree % 16] += resources[temp_node][1]
            node.insert(0, temp_node)
            degree += 1
            continue
    pipeline[degree].append(temp_node)
    last_node = temp_node

# print(degree)

for i in range(0, 607):
    if i > degree:
        pipeline.pop()

# print(pipeline)

# df = pd.DataFrame(pipeline)
# df.to_csv('贪心方案.csv')


# Tweak the optimization based on the greedy scheme, i.e. try to put the current basic block into the previous flow level
changed_flag = 1
current_path_flag = 0
last_path_flag = 0
last_resource_flag = 0
line_copy = []

last_line_flag = 0
last_line = pipeline[0]
while changed_flag != 0:
    changed_flag = 0
    last_line_flag = 0
    last_line = pipeline[0]
    for line in pipeline[1:]:
        # print(line)
        line_copy = line.copy()
        for block in line_copy:
            # Check the path constraint of the current layer
            current_path_flag = 0
            for item in line:
                if correlation[item][block] != 0:
                    current_path_flag = 1
                    break
            # Check the path constraints of the previous layer
            last_path_flag = 0
            for item in last_line:
                if correlation[item][block] == 2:
                    last_path_flag = 1
                    break
            # Check the resource constraints of the previous layer
            last_resource_flag = 0
            if (R_TCAM[last_line_flag] >= resources[block][0]) & (R_HASH[last_line_flag] >= resources[block][1]) \
                    & (R_ALU[last_line_flag] >= resources[block][2]) & (R_QUALIFY[last_line_flag] >= resources[block][3]):
                pass
            else:
                last_resource_flag = 1
            if last_line_flag <= 31:
                if (FOLD_TCAM[last_line_flag % 16] >= resources[block][0]) & \
                        (FOLD_HASH[last_line_flag % 16] >= resources[block][1]):
                    pass
                else:
                    last_resource_flag = 1
            even_TCAM_flag = 0
            if (last_line_flag % 2 == 0) & (resources[block][0] == 1):
                if even_TCAM_count < 5:
                    pass
                else:
                    even_TCAM_flag = 1

            # If all constraints are satisfied, then the current block can be placed in the previous flow level
            if (current_path_flag == 0) & (last_path_flag == 0) & (last_resource_flag == 0) & (even_TCAM_flag == 0):
                # print("Find Optimizable Blocks" + str(block) + ', It is now placed in the' + str(last_line_flag) + 'th pipeline stages')
                changed_flag = 1
                R_TCAM[last_line_flag] -= resources[block][0]
                R_HASH[last_line_flag] -= resources[block][1]
                R_ALU[last_line_flag] -= resources[block][2]
                R_QUALIFY[last_line_flag] -= resources[block][3]
                R_TCAM[last_line_flag+1] += resources[block][0]
                R_HASH[last_line_flag+1] += resources[block][1]
                R_ALU[last_line_flag+1] += resources[block][2]
                R_QUALIFY[last_line_flag+1] += resources[block][3]
                if last_line_flag <= 31:
                    FOLD_TCAM[last_line_flag % 16] -= resources[block][0]
                    FOLD_HASH[last_line_flag % 16] -= resources[block][1]
                if (last_line_flag + 1) <= 31:
                    FOLD_TCAM[(last_line_flag + 1) % 16] += resources[block][0]
                    FOLD_HASH[(last_line_flag + 1) % 16] += resources[block][1]
                if (last_line_flag % 2 == 0) & (resources[block][0] == 1):
                    even_TCAM_count += 1
                if ((last_line_flag + 1) % 2 == 0) & (resources[block][0] == 1):
                    even_TCAM_count -= 1
                last_line.append(block)
                line.remove(block)
            # If only the TCAM even constraint is not satisfied, the current block can be considered to be put into the upper two flow levels
            elif (current_path_flag == 0) & (last_path_flag == 0) & (last_resource_flag == 0) & (even_TCAM_flag == 1):
                last_line_flag -= 1
                # Check the path constraint of the current layer
                current_path_flag = 0
                for item in line:
                    if correlation[item][block] != 0:
                        current_path_flag = 1
                        break
                # Check the path constraint of the previous layer
                last_path_flag = 0
                for item in last_line:
                    if correlation[item][block] == 2:
                        last_path_flag = 1
                        break
                # Check the resource constraints of the previous layer
                last_resource_flag = 0
                if (R_TCAM[last_line_flag] >= resources[block][0]) & (R_HASH[last_line_flag] >= resources[block][1]) \
                        & (R_ALU[last_line_flag] >= resources[block][2]) & (
                        R_QUALIFY[last_line_flag] >= resources[block][3]):
                    pass
                else:
                    last_resource_flag = 1
                if last_line_flag <= 31:
                    if (FOLD_TCAM[last_line_flag % 16] >= resources[block][0]) & \
                            (FOLD_HASH[last_line_flag % 16] >= resources[block][1]):
                        pass
                    else:
                        last_resource_flag = 1
                # If all constraints are satisfied, then the current block can be placed in the last two flow levels
                if (current_path_flag == 0) & (last_path_flag == 0) & (last_resource_flag == 0):
                    # print("Find Optimizable Blocks" + str(block) + ', It is now placed in the' + str(last_line_flag) + 'th pipeline stages')
                    changed_flag = 1
                    pipeline[last_line_flag].append(block)
                    pipeline[last_line_flag + 2].remove(block)
                    R_TCAM[last_line_flag] -= resources[block][0]
                    R_HASH[last_line_flag] -= resources[block][1]
                    R_ALU[last_line_flag] -= resources[block][2]
                    R_QUALIFY[last_line_flag] -= resources[block][3]
                    R_TCAM[last_line_flag + 2] += resources[block][0]
                    R_HASH[last_line_flag + 2] += resources[block][1]
                    R_ALU[last_line_flag + 2] += resources[block][2]
                    R_QUALIFY[last_line_flag + 2] += resources[block][3]
                    if last_line_flag <= 31:
                        FOLD_TCAM[last_line_flag % 16] -= resources[block][0]
                        FOLD_HASH[last_line_flag % 16] -= resources[block][1]
                    if (last_line_flag + 2) <= 31:
                        FOLD_TCAM[(last_line_flag + 2) % 16] += resources[block][0]
                        FOLD_HASH[(last_line_flag + 2) % 16] += resources[block][1]
                last_line_flag += 1
        last_line_flag += 1
        last_line = line

# print(pipeline)

# df = pd.DataFrame(pipeline)
# df.to_csv('optimized_plan.csv')

# print(find_path(307, 33))

resources_left = [[] for _ in range(4)]
resources_left[0] = R_TCAM[0:100]
resources_left[1] = R_HASH[0:100]
resources_left[2] = R_ALU[0:100]
resources_left[3] = R_QUALIFY[0:100]

df = pd.DataFrame(resources_left)
df.to_csv('all_levels_of_stage_remaining_resources.csv')

resources_1 = [[] for _ in range(4)]
for i in resources_left[0]:
    resources_1[0].append(1 - i)
for i in resources_left[1]:
    resources_1[1].append(2 - i)
for i in resources_left[2]:
    resources_1[2].append(56 - i)
for i in resources_left[3]:
    resources_1[3].append(64 - i)
df = pd.DataFrame(resources_1)
df.to_csv('all_levels_of_stage_using_resources_question_one.csv')

# A scheduling scheme for the second problem is given in a similar way to problem 1
node = list(range(0, 607))
node.sort(reverse=False, key=lambda x: path_depth[x])
# print(node)

R_TCAM_2 = [0] * 607
R_HASH_2 = [0] * 607
R_ALU_2 = [0] * 607
R_QUALIFY_2 = [0] * 607
FOLD_TCAM_2 = [0] * 16
FOLD_HASH_2 = [0] * 16
even_TCAM_count_2 = 0
path_flag = 0

pipeline_2 = [[] for _ in range(607)]
degree = 0
hash_level = 0
alu_level = 0
fold_hash_level = 0
last_node = node[0]
node_hash = [resources[i][1] for i in range(0, 607)]
node_alu = [resources[i][2] for i in range(0, 607)]
node_fold_hash = node_hash.copy()

# Start formally placing nodes into the pipeline
while len(node) != 0:
    temp_node = node[0]
    put_flag = 0
    # Check path constraints
    if (int(path_depth[temp_node]) - int(path_depth[last_node]) >= 1) & (path_flag == 0):
        # print("put node" + str(temp_node) + "failed, running segment" + str(tree) + "due to path constraint, previous node is " + str(last_node), path_depth[temp_node], path_depth[last_node])
        degree += 1
        path_flag = 1
    # Check TCAM constraints
    if R_TCAM_2[degree] + resources[temp_node][0] <= 1:
        pass
    else:
        put_flag = 1
    # Check QUALIFY constraints
    if R_QUALIFY_2[degree] + resources[temp_node][3] <= 64:
        pass
    else:
        put_flag = 1
    # Check HASH constraint, need to consider the internal flow relationship of the flow segment
    hash_level = node_hash[temp_node]
    for block in pipeline_2[degree]:
        if find_path_2(block, temp_node):
            if hash_level < node_hash[block] + node_hash[temp_node]:
                hash_level = node_hash[block] + node_hash[temp_node]
    if hash_level <= 2:
        pass
    else:
        put_flag = 1
    # Check ALU constraint, need to consider the internal flow relationship of the flow segment
    alu_level = node_alu[temp_node]
    for block in pipeline_2[degree]:
        if find_path_2(block, temp_node):
            if alu_level < node_alu[block] + node_alu[temp_node]:
                alu_level = node_alu[block] + node_alu[temp_node]
    if alu_level <= 56:
        pass
    else:
        put_flag = 1
    # Check for collapse constraints
    if degree <= 31:
        # Check for collapsing TCAM constraints
        if FOLD_TCAM_2[degree % 16] + resources[temp_node][0] <= 1:
            pass
        else:
            put_flag = 1
        # Check the collapsed HASH constraint, need to consider the internal flow relationship of the flow segment
        fold_hash_level = node_fold_hash[temp_node]
        for block in pipeline_2[degree]:
            if find_path_2(block, temp_node):
                if fold_hash_level < node_fold_hash[block] + node_fold_hash[temp_node]:
                    fold_hash_level = node_fold_hash[block] + node_fold_hash[temp_node]
        if fold_hash_level <= 3:
            pass
        else:
            put_flag = 1
    # Check TCAM even level limits
    if (degree % 2 == 0) & (resources[temp_node][0] == 1):
        if even_TCAM_count_2 < 5:
            pass
        else:
            put_flag = 1

    # If there are constraints that are not satisfied, consider putting in the next level of flow
    if put_flag == 1:
        # print("放入节点" + str(temp_node) + "失败,流水段" + str(degree) + "...........................")
        degree += 1
        continue
    # If all constraints are satisfied, start putting in nodes
    else:
        node.pop(0)
        last_node = temp_node
        pipeline_2[degree].append(temp_node)
        path_flag = 0
        # print("放入节点" + str(temp_node) + ",流水段" + str(degree))
        R_TCAM_2[degree] += resources[temp_node][0]
        R_HASH_2[degree] = max(R_HASH_2[degree], hash_level)
        R_ALU_2[degree] = max(R_ALU_2[degree], alu_level)
        R_QUALIFY_2[degree] += resources[temp_node][3]
        node_hash[temp_node] = hash_level
        node_alu[temp_node] = alu_level
        if degree <= 31:
            FOLD_TCAM_2[degree % 16] += resources[temp_node][0]
            node_fold_hash[temp_node] = fold_hash_level
            FOLD_HASH_2[degree % 16] = max(FOLD_HASH_2[degree % 16], fold_hash_level)
        if (degree % 2 == 0) & (resources[temp_node][0] == 1):
            even_TCAM_count_2 += 1

# print(pipeline_2)
# df = pd.DataFrame(pipeline_2)
# df.to_csv('greedy_program_problem_2.csv')

# print(resources[90])

# Tune the initial solution to problem 2
changed_flag = 1
current_path_flag = 0
last_path_flag = 0
last_resource_flag = 0
line_copy = []

last_line_flag = 0
last_line = pipeline_2[0]
while changed_flag != 0:
    changed_flag = 0
    last_line_flag = 0
    last_line = pipeline_2[0]
    for line in pipeline_2[1:]:
        # print(line)
        line_copy = line.copy()
        for block in line_copy:
            put_flag = 0
            even_TCAM_flag = 0
            # Check the path constraint of the current layer
            current_path_flag = 0
            for item in line:
                if correlation[item][block] != 0:
                    put_flag = 1
                    break
            # Check the path constraints of the previous layer
            last_path_flag = 0
            for item in last_line:
                if correlation[item][block] == 2:
                    put_flag = 1
                    break
            # Check the resource constraints of the previous layer
            # Check TCAM constraints
            if R_TCAM_2[last_line_flag] + resources[block][0] <= 1:
                pass
            else:
                put_flag = 1
            # Check QUALIFY constraints
            if R_QUALIFY_2[last_line_flag] + resources[block][3] <= 64:
                pass
            else:
                put_flag = 1
            # Check HASH constraint, need to consider the internal flow relationship of the flow segment
            hash_level = resources[block][1]
            for block_1 in pipeline_2[last_line_flag]:
                if find_path_2(block_1, block):
                    if hash_level < node_hash[block_1] + node_hash[block]:
                        hash_level = node_hash[block_1] + node_hash[block]
            if hash_level <= 2:
                pass
            else:
                put_flag = 1
            # Check ALU constraint, need to consider the internal flow relationship of the flow segment
            alu_level = resources[block][2]
            for block_1 in pipeline_2[last_line_flag]:
                if find_path_2(block_1, block):
                    if alu_level < node_alu[block] + node_alu[block_1]:
                        alu_level = node_alu[block] + node_alu[block_1]
            if alu_level <= 56:
                pass
            else:
                put_flag = 1
            # Check for collapse constraints
            if last_line_flag <= 31:
                # Check for collapsing TCAM constraints
                if FOLD_TCAM_2[last_line_flag % 16] + resources[block][0] <= 1:
                    pass
                else:
                    put_flag = 1
                # Check the collapsed HASH constraint, need to consider the internal flow relationship of the flow segment
                fold_hash_level = resources[block][1]
                for block_1 in pipeline_2[last_line_flag]:
                    if find_path_2(block_1, block):
                        if fold_hash_level < node_fold_hash[block] + node_fold_hash[block_1]:
                            fold_hash_level = node_fold_hash[block] + node_fold_hash[block_1]
                if fold_hash_level <= 3:
                    pass
                else:
                    put_flag = 1
            # Check TCAM even level limits
            if (last_line_flag % 2 == 0) & (resources[block][0] == 1):
                if even_TCAM_count_2 < 5:
                    pass
                else:
                    even_TCAM_flag = 1

            # If all constraints are satisfied, then the current block can be placed in the previous flow level
            if (put_flag == 0) & (even_TCAM_flag == 0):
                # Put the current block into the previous pipeline level
                # print("Found an optimizable block" + str(block) + ', now put it into the ' + str(last_line_flag) + 'level pipeline segment')
                changed_flag = 1
                R_TCAM_2[last_line_flag] += resources[block][0]
                R_QUALIFY_2[last_line_flag] += resources[block][3]
                R_HASH_2[last_line_flag] = max(R_HASH_2[last_line_flag], hash_level)
                R_ALU_2[last_line_flag] = max(R_ALU_2[last_line_flag], alu_level)
                node_hash[block] = hash_level
                node_alu[block] = alu_level
                if last_line_flag <= 31:
                    FOLD_TCAM_2[last_line_flag % 16] += resources[block][0]
                    node_fold_hash[block] = fold_hash_level
                    FOLD_HASH_2[last_line_flag % 16] = max(FOLD_HASH_2[last_line_flag % 16], fold_hash_level)
                if (last_line_flag % 2 == 0) & (resources[block][0] == 1):
                    even_TCAM_count_2 += 1
                last_line.append(block)

                # Move the current block out of the current flow level
                line.remove(block)
                R_TCAM_2[last_line_flag+1] -= resources[block][0]
                R_QUALIFY_2[last_line_flag+1] -= resources[block][3]
                hash_level = 0
                alu_level = 0
                temp_line = []
                R_HASH_2[last_line_flag + 1] = 0
                R_ALU_2[last_line_flag + 1] = 0
                for item in line:
                    node_hash[item] = resources[item][1]
                    hash_level = node_hash[item]
                    node_alu[item] = resources[item][2]
                    alu_level = node_alu[item]
                    for item_1 in temp_line:
                        if find_path_2(item_1, item):
                            if hash_level < node_hash[item_1] + node_hash[item]:
                                hash_level = node_hash[item_1] + node_hash[item]
                            if alu_level < node_alu[item] + node_alu[item_1]:
                                alu_level = node_alu[item] + node_alu[item_1]
                    node_hash[item] = hash_level
                    node_alu[item] = alu_level
                    R_HASH_2[last_line_flag+1] = max(R_HASH_2[last_line_flag+1], hash_level)
                    R_ALU_2[last_line_flag+1] = max(R_ALU_2[last_line_flag+1], alu_level)
                    temp_line.append(item)
                if (last_line_flag + 1) <= 31:
                    FOLD_TCAM_2[(last_line_flag + 1) % 16] -= resources[block][0]
                    FOLD_HASH_2[(last_line_flag + 1) % 16] = R_HASH_2[(last_line_flag+1) % 16]\
                        + R_HASH_2[(last_line_flag+1) % 16 + 16]
                if ((last_line_flag + 1) % 2 == 0) & (resources[block][0] == 1):
                    even_TCAM_count_2 -= 1
            # If limited by TCAM even levels only, consider putting it in the first two levels
            elif (put_flag == 0) & (even_TCAM_flag == 1):
                last_line_flag -= 1
                put_flag = 0
                # Check the path constraint of the current layer
                for item in line:
                    if correlation[item][block] != 0:
                        put_flag = 1
                        break
                # Check the path constraints of the previous layer
                for item in last_line:
                    if correlation[item][block] == 2:
                        put_flag = 1
                        break
                # Check the resource constraints of the previous layer
                # Check TCAM constraints
                if R_TCAM_2[last_line_flag] + resources[block][0] <= 1:
                    pass
                else:
                    put_flag = 1
                # Check QUALIFY constraints
                if R_QUALIFY_2[last_line_flag] + resources[block][3] <= 64:
                    pass
                else:
                    put_flag = 1
                # Check HASH constraint, need to consider the internal flow relationship of the flow segment
                hash_level = resources[block][1]
                for block_1 in pipeline_2[last_line_flag]:
                    if find_path_2(block_1, block):
                        if hash_level < node_hash[block_1] + node_hash[block]:
                            hash_level = node_hash[block_1] + node_hash[block]
                if hash_level <= 2:
                    pass
                else:
                    put_flag = 1
                # Check ALU constraint, need to consider the internal flow relationship of the flow segment
                alu_level = resources[block][2]
                for block_1 in pipeline_2[last_line_flag]:
                    if find_path_2(block_1, block):
                        if alu_level < node_alu[block] + node_alu[block_1]:
                            alu_level = node_alu[block] + node_alu[block_1]
                if alu_level <= 56:
                    pass
                else:
                    put_flag = 1
                # Check for collapse constraints
                if last_line_flag <= 31:
                    # Check for collapsing TCAM constraints
                    if FOLD_TCAM_2[last_line_flag % 16] + resources[block][0] <= 1:
                        pass
                    else:
                        put_flag = 1
                    # Check the collapsed HASH constraint, need to consider the internal flow relationship of the flow segment
                    fold_hash_level = resources[block][1]
                    for block_1 in pipeline_2[last_line_flag]:
                        if find_path_2(block_1, block):
                            if fold_hash_level < node_fold_hash[block] + node_fold_hash[block_1]:
                                fold_hash_level = node_fold_hash[block] + node_fold_hash[block_1]
                    if fold_hash_level <= 3:
                        pass
                    else:
                        put_flag = 1
                # If all constraints are satisfied, then the current block can be placed in the last two flow levels
                if put_flag == 0:
                    # Put the current block into the last two pipeline levels
                    # print("Found an optimizable block" + str(block) + ', now put it into the first ' + str(last_line_flag) + 'level of the pipeline')
                    changed_flag = 1
                    R_TCAM_2[last_line_flag] += resources[block][0]
                    R_QUALIFY_2[last_line_flag] += resources[block][3]
                    R_HASH_2[last_line_flag] = max(R_HASH_2[last_line_flag], hash_level)
                    R_ALU_2[last_line_flag] = max(R_ALU_2[last_line_flag], alu_level)
                    node_hash[block] = hash_level
                    node_alu[block] = alu_level
                    if last_line_flag <= 31:
                        FOLD_TCAM_2[last_line_flag % 16] += resources[block][0]
                        node_fold_hash[block] = fold_hash_level
                        FOLD_HASH_2[last_line_flag % 16] = max(FOLD_HASH_2[last_line_flag % 16],
                                                               fold_hash_level)
                    pipeline_2[last_line_flag].append(block)

                    # Move the current block out of the current flow level
                    line.remove(block)
                    R_TCAM_2[last_line_flag + 2] -= resources[block][0]
                    R_QUALIFY_2[last_line_flag + 2] -= resources[block][3]
                    hash_level = 0
                    alu_level = 0
                    temp_line = []
                    R_HASH_2[last_line_flag + 2] = 0
                    R_ALU_2[last_line_flag + 2] = 0
                    for item in line:
                        node_hash[item] = resources[item][1]
                        hash_level = node_hash[item]
                        node_alu[item] = resources[item][2]
                        alu_level = node_alu[item]
                        for item_1 in temp_line:
                            if find_path_2(item_1, item):
                                if hash_level < node_hash[item_1] + node_hash[item]:
                                    hash_level = node_hash[item_1] + node_hash[item]
                                if alu_level < node_alu[item] + node_alu[item_1]:
                                    alu_level = node_alu[item] + node_alu[item_1]
                        node_hash[item] = hash_level
                        node_alu[item] = alu_level
                        R_HASH_2[last_line_flag + 2] = max(R_HASH_2[last_line_flag + 2], hash_level)
                        R_ALU_2[last_line_flag + 2] = max(R_ALU_2[last_line_flag + 2], alu_level)
                        temp_line.append(item)
                    if (last_line_flag + 2) <= 31:
                        FOLD_TCAM_2[(last_line_flag + 2) % 16] -= resources[block][0]
                        FOLD_HASH_2[(last_line_flag + 2) % 16] = R_HASH_2[(last_line_flag + 2) % 16] \
                                                                 + R_HASH_2[(last_line_flag + 2) % 16 + 16]
                last_line_flag += 1
        last_line_flag += 1
        last_line = line

# df = pd.DataFrame(pipeline_2)
# df.to_csv('Optimization_solution.csv')

resources_2 = [[] for _ in range(4)]
resources_2[0] = R_TCAM_2[0:51]
resources_2[1] = R_HASH_2[0:51]
resources_2[2] = R_ALU_2[0:51]
resources_2[3] = R_QUALIFY_2[0:51]

# df = pd.DataFrame(resources_2)
# df.to_csv('Resource_schedule.csv')
