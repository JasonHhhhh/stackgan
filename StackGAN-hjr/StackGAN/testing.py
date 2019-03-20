# author: Jason Howe
cap_path = 'F:\Jason Howe\Program\hjr\StackGAN-master\StackGAN\Data/birds/text_c10/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.txt'
with open(cap_path, "r") as f:
    captions = f.read().split('\n')
print(captions)
captions = [cap for cap in captions if len(cap) > 0]
print(captions)

