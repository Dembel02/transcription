import matplotlib.pyplot as plt

# Данные для графика
categories = ['Доход', 'Затраты', 'Прибыль']
values = [1200000, 242000, 958000]

plt.bar(categories, values, color=['green', 'red', 'blue'])
plt.title('Финансовый анализ аренды долот')
plt.ylabel('Сумма в долларах')
plt.show()
