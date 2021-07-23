import random

def find(lines, string):
	obj = []
	for i in range(len(lines)):
		if lines[i].find(string) != -1:
			obj.append(lines[i])
			continue
	return obj

def find_label(src, dst):
	target = []
	for i in range(len(dst)):
		name = dst[i][0]
		for j in range(len(src)):
			if src[j].find(name) != -1:
				target.append(src[j])
				break
			if j == len(src)-1:
				print('wrong with pairs!')
	return target


def write_into_file(flag, lines, file):
	if flag:
		for line in lines:
			file.write(line[0]+' '+str(line[1])+'\n')
	else:
		for line in lines:
			file.write(line)


def create_pair(train_list, loops):
	obj = []
	for k in range(loops):
		for i in range(len(train_list)):
			x = train_list[i].split(' ')
			score = float(x[1].split("\r")[0])
			random.seed()
			j = random.randint(0, len(train_list)-1)
			x_p = train_list[j].split(' ')
			score_p = float(x_p[1].split("\r")[0])

			while abs(score_p - score) > 1 or score_p == score:
				j = random.randint(0, len(train_list)-1)
				x_p = train_list[j].split(' ')
				score_p = float(x_p[1].split("\r")[0])
			prob = random.randint(0, 9)

			if score - score_p > 0:
				if prob <= 5:
					obj.append((x[0], x_p[0], 1))
				else:
					obj.append((x_p[0], x[0], -1))

			elif score - score_p < 0:
				if prob <= 5:
					obj.append((x[0], x_p[0], -1))
				else:			
					obj.append((x_p[0], x[0], 1))
			else:
				print('wrong!')
		print('iteration number: %d'%(k+1))

	return obj


def main():
	filename = "./1/train_1.txt"
	source = open(filename, "r")
	train_list = source.readlines()
	source.close()

	# create pairs accoding to males and females
	male = find(train_list, 'mt')
	print('male number: %d' %len(male))
	male_list = create_pair(male, loops=25)
	print('male pairs number: %d' %len(male_list))

	female = find(train_list, 'ft')
	print('female number: %d' %len(female))
	female_list = create_pair(female, loops=25)
	print('female pairs number: %d' %len(female_list))

	# merge and shuffle pairs
	male_list.extend(female_list)
	pairs = male_list
	random.shuffle(pairs)

	# split pairs
	x = [(pairs[i][0], pairs[i][2]) for i in range(len(pairs))]
	x_p = [(pairs[i][1], pairs[i][2]) for i in range(len(pairs))]

	# create single label file corresponding to pair files	
	single = find_label(src=train_list, dst=x)
	single_p = find_label(src=train_list, dst=x_p)

	# write to files
	x_file = open("./1/train.txt", 'w')
	write_into_file(flag=True, lines=x, file=x_file)
	x_file.close()

	x_p_file = open("./1/train_p.txt", 'w')
	write_into_file(flag=True, lines=x_p, file=x_p_file)
	x_p_file.close()

	single_file = open('./1/single.txt', 'w')
	write_into_file(flag=False, lines=single, file=single_file)
	single_file.close()

	single_p_file = open('./1/single_p.txt', 'w')
	write_into_file(flag=False, lines=single_p, file=single_p_file)
	single_p_file.close()

	print 'Done!'

if __name__ == '__main__':
	main()
