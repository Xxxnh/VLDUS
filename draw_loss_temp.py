import re
import matplotlib.pyplot as plt

loss_dict = {
    'Loss': [],
    'D loss': [],
    'G loss': [],
    'W dist': [],
    'seen loss': [],
    'loss div': []
}

with open('output.log', 'r') as file:
    for line in file:
        match = re.search(r'Loss: (.+?) D loss: (.+?) G loss: (.+?), W dist: (.+?) seen loss: (.+?) loss div: (.+)', line)
        if match:
            loss_dict['Loss'].append(float(match.group(1)))
            loss_dict['D loss'].append(float(match.group(2)))
            loss_dict['G loss'].append(float(match.group(3)))
            loss_dict['W dist'].append(float(match.group(4)))
            loss_dict['seen loss'].append(float(match.group(5)))
            loss_dict['loss div'].append(float(match.group(6)))

# Plotting the loss values
x = range(1, len(loss_dict['G loss']) + 1)


plt.plot(x, loss_dict['G loss'], label='fake')
result = [x + y for x, y in zip(loss_dict['W dist'], loss_dict['G loss'])]
plt.plot(x, result, label='real')
plt.plot(x, loss_dict['W dist'], label='real-fake')



plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('loss_temp_f_r.png')