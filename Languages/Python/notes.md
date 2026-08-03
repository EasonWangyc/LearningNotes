# 面向对象

Python 是一门面向对象的编程语言。所有的 Python 对象和数据结构都包含在类中。

## 类和对象

类（Class）是创建对象的蓝图，对象（Object）是类的实例。
- `class` 关键字用于定义类。
- `__init__` 是构造方法，在创建对象时自动调用。
- `self` 代表类的实例，用于访问属性和方法。

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

# 创建对象
p1 = Person("Alice", 30)
p1.greet()
```

## 继承

允许我们定义继承另一个类所有方法和属性的类。
- **父类**（Base Class）：被继承的类。
- **子类**（Derived Class）：继承的类。
- `super()` 函数用于调用父类的方法。

```python
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age) # 调用父类的构造方法
        self.student_id = student_id

    def study(self):
        print(f"{self.name} is studying.")

s1 = Student("Bob", 20, "S12345")
s1.greet()  # 继承自 Person
s1.study()  # Student 自己的方法
```

## 多态

多态是指不同类的对象对同一消息作出不同的响应。在 Python 中，这意味着即使不知道变量所引用的对象类型，也可以对它进行操作，只要它具有该操作的方法。

```python
class Teacher(Person):
    def greet(self):
        print(f"Good morning, I am {self.name}, your teacher.")

def introduce(person):
    person.greet()

p = Person("Charlie", 40)
s = Student("David", 19, "S67890")
t = Teacher("Eva", 35)

introduce(p)
introduce(s)
introduce(t)
```

# 数据结构

## list

## dict


## tuple
