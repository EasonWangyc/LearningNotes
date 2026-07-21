## 简介
C++ 是一种静态类型的、编译式的、通用的、大小写敏感的、不规则的编程语言，支持过程化编程、面向对象编程和泛型编程。C++ 被认为是一种中级语言，它综合了高级语言和低级语言的特点。C++ 是 C 的一个超集，事实上，任何合法的 C 程序都是合法的 C++ 程序。

## 数据类型
数据类型与C语言类型，只不过增加了bool、wchar_t、long long等类型。

wchar_t是一种宽字符类型，通常用于表示Unicode字符。它的大小通常为2或4字节，具体取决于平台和编译器。wchar_t可以存储更多的字符集，包括亚洲语言和其他非拉丁字符。
```cpp
typedef short int wchar_t;
```
### 占用内存
|数据类型|描述|大小（字节）|范围/取值示例|
|---|---|---|---|
|`bool`|布尔类型，表示真或假|1|`true` 或 `false`|
|`char`|字符类型，通常用于存储 ASCII 字符|1|\-128 到 127 或 0 到 255|
|`signed char`|有符号字符类型|1|\-128 到 127|
|`unsigned char`|无符号字符类型|1|0 到 255|
|`wchar_t`|宽字符类型，用于存储 Unicode 字符|2 或 4|取决于平台|
|`char16_t`|16 位 Unicode 字符类型（C++11 引入）|2|0 到 65,535|
|`char32_t`|32 位 Unicode 字符类型（C++11 引入）|4|0 到 4,294,967,295|
|`short`|短整型|2|\-32,768 到 32,767|
|`unsigned short`|无符号短整型|2|0 到 65,535|
|`int`|整型|4|\-2,147,483,648 到 2,147,483,647|
|`unsigned int`|无符号整型|4|0 到 4,294,967,295|
|`long`|长整型|4 或 8|取决于平台|
|`unsigned long`|无符号长整型|4 或 8|取决于平台|
|`long long`|长长整型（C++11 引入）|8|\-9,223,372,036,854,775,808 到 9,223,372,036,854,775,807|
|`unsigned long long`|无符号长长整型（C++11 引入）|8|0 到 18,446,744,073,709,551,615|
|`float`|单精度浮点数|4|约 ±3.4e±38（6-7 位有效数字）|
|`double`|双精度浮点数|8|约 ±1.7e±308（15 位有效数字）|
|`long double`|扩展精度浮点数|8、12 或 16|取决于平台|

### 修饰符

|修饰符|描述|示例|
|---|---|---|
|`signed`|表示有符号类型（默认）|`signed int x = -10;`|
|`unsigned`|表示无符号类型|`unsigned int y = 10;`|
|`short`|表示短整型|`short int z = 100;`|
|`long`|表示长整型|`long int a = 100000;`|

### 类型限定符
|限定符|含义|示例|
|---|---|---|
|`const`|表示常量，值不可修改|`const int b = 5;`|
|`volatile`|表示变量可能被意外修改，禁止编译器优化|`volatile int c = 10;`|
|`restrict`|表示指针是唯一访问对象的方式（C99 引入）|`int* restrict ptr;`|
|`mutable`|表示类成员可以在 `const` 对象中修改|`mutable int counter;`|
|`static`|表示变量具有静态存储持续时间|`static int count = 0;`|

### 类型转换

#### 静态类型转换
静态转换（`static_cast`）是将一种数据类型的值强制转换为另一种数据类型的值。静态转换通常用于比较类型相似的对象之间的转换，例如将 int 类型转换为 float 类型。静态转换不进行任何运行时类型检查，因此可能会导致运行时错误。
```cpp
int i = 10; 
float f = static_cast<float>(i);
```
#### 动态类型转换
动态转换（`dynamic_cast`）是 C++ 中用于在继承层次结构中进行向下转换（downcasting）的一种机制。动态转换通常用于将一个基类指针或引用转换为派生类指针或引用。

动态转换在运行时进行类型检查。如果转换失败，对于指针类型会返回 nullptr，对于引用类型则会抛出 `std::bad_cast`异常。

语法：
```cpp
dynamic_cast<new_type>(expression)
```
其中：
- `new_type` 是要转换成的类型，必须是一个指针或引用类型。
- `expression` 是要转换的表达式，必须是一个指针或引用类型，并且必须指向一个多态类型（即至少有一个虚函数的类）。

示例：
```cpp
// 指针类型的动态转换
#include <iostream> 
class Base { 
    public: virtual ~Base() = default; 
}; 
class Derived : public Base { 
public: 
    void show() { 
        std::cout << "Derived class method" << std::endl; 
        } 
}; 
int main() { 
    Base* ptr_base = new Derived; 
    Derived* ptr_derived = dynamic_cast<Derived*>(ptr_base); 
    if (ptr_derived) { 
        ptr_derived->show(); 
    } 
    else { 
        std::cout << "Dynamic cast failed!" << std::endl; 
    } 
    delete ptr_base; 
    return 0; 
}
```
```cpp
// 应用类型的动态转换
#include <iostream>
#include <typeinfo>

class Base {
public:
    virtual ~Base() = default; // 基类必须具有虚函数
};

class Derived : public Base {
public:
    void show() {
        std::cout << "Derived class method" << std::endl;
    }
};

int main() {
    Derived derived_obj;
    Base& ref_base = derived_obj; // 基类引用绑定到派生类对象
    try {
        // 将基类引用转换为派生类引用
        Derived& ref_derived = dynamic_cast<Derived&>(ref_base);
        ref_derived.show(); // 成功转换，调用派生类方法
    } catch (const std::bad_cast& e) {
        std::cout << "Dynamic cast failed: " << e.what() << std::endl;
    }
    return 0;
}
```
#### 常量类型转换
常量转换用于将 const 类型的对象转换为非 const 类型的对象。常量转换只能用于转换掉 const 属性，不能改变对象的类型。

#### 重新解释类型转换
重新解释转换将一个数据类型的值重新解释为另一个数据类型的值，通常用于在不同的数据类型之间进行转换。重新解释转换不进行任何类型检查，因此可能会导致未定义的行为。

## 数据结构
### Array数组
数组是最基础的数据结构，用于存储一组相同类型的数据。

特点：
- 固定大小，一旦声明，大小不能改变。
- 直接访问元素，时间复杂度为 O(1)。
- 适合处理大小已知、元素类型相同的集合。

优缺点：
- 优点：访问速度快，内存紧凑。
- 缺点：大小固定，无法动态扩展，不适合处理大小不确定的数据集。

### Vector动态数组
C++ 中的 vector 是一种序列容器，它允许你在运行时动态地插入和删除元素。vector 是基于数组的数据结构，但它可以自动管理内存，这意味着你不需要手动分配和释放内存。与 C++ 数组相比，vector 具有更多的灵活性和功能，使其成为 C++ 中常用的数据结构之一。vector 是 C++ 标准模板库（STL）的一部分，提供了灵活的接口和高效的操作。

特点:
- 动态大小：vector 的大小可以根据需要自动增长和缩小。
- 连续存储：vector 中的元素在内存中是连续存储的，这使得访问元素非常快速。
- 可迭代：vector 可以被迭代，你可以使用循环（如 for 循环）来访问它的元素。
- 元素类型：vector 可以存储任何类型的元素，包括内置类型、对象、指针等。

使用场景：
- 当你需要一个可以动态增长和缩小的数组时。
- 当你需要频繁地在序列的末尾添加或移除元素时。
- 当你需要一个可以高效随机访问元素的容器时。

创建vector需要先引入vector头文件。
```cpp
#include <iostream>
#include <vector>
```
创建 Vector：
```cpp
std::vector<int> myVector1; // 创建一个存储整数的空 vector
std::vector<int> myVector2(5); // 创建一个包含 5 个整数的 vector，每个值都为默认值（0）
std::vector<int> myVector3(5, 10); // 创建一个包含 5 个整数的 vector，每个值都为 10
std::vector<int> myVector4 = {1, 2, 3, 4}; // 初始化一个包含元素的 vector
```
可以使用 `push_back` 方法向 vector 中添加元素：
```cpp
myVector.push_back(7); // 将整数 7 添加到 vector 的末尾
```
可以使用下标操作符 `[]` 或 `at()` 方法访问 vector 中的元素：
```cpp
int x = myVector[0]; // 获取第一个元素
int y = myVector.at(1); // 获取第二个元素
```
可以使用` size()` 方法获取 vector 中元素的数量：
```cpp
int size = myVector.size(); // 获取 vector 中的元素数量
```
可以使用迭代器遍历 vector 中的元素：
```cpp
for (auto it = myVector.begin(); it != myVector.end(); ++it) {
    std::cout << *it << " ";
}
```
或者使用范围循环：
```cpp
for (int element : myVector) {
    std::cout << element << " ";
}
```
可以使用 `erase()` 方法删除 vector 中的元素：
```cpp
myVector.erase(myVector.begin() + 2); // 删除第三个元素
```
可以使用 `clear()` 方法清空 vector 中的所有元素：
```cpp
myVector.clear(); // 清空 vector
```
可以使用`insert()` 方法在 vector 中插入元素：
```cpp
myVector.insert(myVector.begin() + 2, 99); // 在第三个位置插入元素 99
```

### Struct结构体
结构体允许将不同类型的数据组合在一起，形成一种自定义的数据类型。

特点：
- 可以包含不同类型的成员变量。
- 提供了对数据的基本封装，但功能有限。
示例：
```cpp
struct Person {
    string name;
    int age;
};
Person p = {"Alice", 25};
cout << p.name << endl; // 输出 Alice
```

### Pair组
C++ 中的 `std::pair` 是定义在 <utility> 头文件中的一个类模板，用于将两个可能类型不同的数据组合成一个单一的单元（键值对）。主要成员为公有的 `first` 和 `second`。常用 `std::make_pair` 简化创建，特别适用于函数返回多个值或在 `std::map` 中存储数。本质上是一个Struct结构体。
```cpp
#include <iostream>
#include <utility>
#include <string>

int main() {
    // 1. 定义和初始化
    std::pair<std::string, int> p1("Age", 20);
    std::pair<std::string, int> p2 = std::make_pair("Score", 95); // 使用 make_pair

    // 2. 访问元素
    std::cout << p1.first << ": " << p1.second << std::endl; // 输出: Age: 20
    
    return 0;
}

```

### Class类
类是 C++ 中用于面向对象编程的核心结构，允许定义成员变量和成员函数。与 struct 类似，但功能更强大，支持继承、封装、多态等特性。

特点：
- 可以包含成员变量、成员函数、构造函数、析构函数。
- 支持面向对象特性，如封装、继承、多态。
```cpp
class Person {
private:
    string name;
    int age;
public:
    Person(string n, int a) : name(n), age(a) {}
    void printInfo() {
        cout << "Name: " << name << ", Age: " << age << endl;
    }
};
Person p("Bob", 30);
p.printInfo(); // 输出: Name: Bob, Age: 30
```

### Linked List链表
链表是一种动态数据结构，由一系列节点组成，每个节点包含数据和指向下一个节点的指针。

特点：
- 动态调整大小，不需要提前定义容量。
- 插入和删除操作效率高，时间复杂度为 O(1)（在链表头部或尾部操作）。
- 线性查找，时间复杂度为 O(n)。
  
优缺点：

- 优点：动态大小，适合频繁插入和删除的场景。
- 缺点：随机访问效率低，不如数组直接访问快。

示例（单向链表）
```cpp
struct Node {
    int data;   // 节点数据
    Node* next; // 指向下一个节点的指针
};
Node* head = nullptr;
Node* newNode = new Node{10, nullptr};
head = newNode; // 插入新节点
```

### Stack栈
栈是一种后进先出（LIFO, Last In First Out）的数据结构，常用于递归、深度优先搜索等场景。

特点：
- 只允许在栈顶进行插入和删除操作。
- 时间复杂度为 O(1)。

优缺点：
- 优点：操作简单，效率高。
- 缺点：只能在栈顶操作，访问其他元素需要弹出栈顶元素。
  
```cpp
stack<int> st;
st.push(1);         // 入栈 1
st.top();           // 获取栈顶元素
st.pop();           // 出栈
st.empty();         // 判断栈是否为空
st.size();          // 获取栈的大小
st.emplace(2);      // 直接在栈顶构造元素
```

### Heap堆
堆是一种先入先出（FIFO）的数据结构，常用于

特点：
- 堆

小顶堆

大顶堆

优先队列

最小堆和最大堆

### Queue队列
队列是一种先进先出（FIFO, First In First Out）的数据结构，常用于广度优先搜索、任务调度等场景。

特点:
- 插入操作在队尾进行，删除操作在队头进行。
- 时间复杂度为 O(1)。

优缺点：
- 优点：适合按顺序处理数据的场景，如任务调度。
- 缺点：无法随机访问元素。
```cpp
queue<int> q;
q.push(1);          // [1]
q.push(2);          // [1,2]
cout << q.front();  // 输出 1
q.pop();            // 移除队头的1
cout << q.front();  // 输出 2
```

### Deque双端队列
双端队列允许在两端进行插入和删除操作，是栈和队列的结合体。

特点：
- 允许在两端进行插入和删除。
- 时间复杂度为 O(1)。
  
优缺点：
- 优点：灵活的双向操作。
- 缺点：空间占用较大，适合需要在两端频繁操作的场景。
```cpp
deque<int> dq;
dq.push_back(1);
dq.push_front(2);
cout << dq.front(); // 输出 2
dq.pop_front();
```

### Hash Table哈希表
哈希表是一种通过键值对存储数据的数据结构，支持快速查找、插入和删除操作。C++ 中的 `unordered_map` 是哈希表的实现。
特点：
- 使用哈希函数快速定位元素，时间复杂度为 O(1)。
- 不保证元素的顺序。

优缺点：
- 优点：查找、插入、删除操作效率高。
- 缺点：无法保证元素顺序，哈希冲突时性能会下降。
  
```cpp
unordered_map<string, int> hashTable;
hashTable["apple"] = 10;
cout << hashTable["apple"]; // 输出 10
```

### Binary Tree二叉树
二叉树是一种树形数据结构，每个节点最多有两个子节点，分别称为左子节点和右子节点。二叉树广泛应用于各种算法和数据结构中，如二叉搜索树、堆等。
特点：
- 每个节点最多有两个子节点。
- 适合表示层次结构的数据。
- 常用于实现高效的搜索和排序算法。
  
优缺点：
- 优点：结构清晰，适合表示层次关系。
- 缺点：不适合存储大量数据，可能导致树的高度增加。

二叉树的遍历方式：
- 前序遍历（Pre-order Traversal）：访问根节点 -> 访问左子树 -> 访问右子树。
- 中序遍历（In-order Traversal）：访问左子树 -> 访问根节点 -> 访问右子树。
- 后序遍历（Post-order Traversal）：访问左子树 -> 访问右子树 -> 访问根节点。

```cpp
#include <iostream>
using namespace std;

// 1. 定义二叉树节点结构
struct TreeNode {
    int data;
    TreeNode* left;
    TreeNode* right;

    // 构造函数
    TreeNode(int val) : data(val), left(nullptr), right(nullptr) {}
};

// 2. 遍历算法：前序遍历（根-左-右）
void preorderTraversal(TreeNode* root) {
    if (root == nullptr) return;
    cout << root->data << " ";
    preorderTraversal(root->left);
    preorderTraversal(root->right);
}

int main() {
    // 3. 手动创建一棵二叉树
    //      1
    //     / \
    //    2   3
    //   / \
    //  4   5
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);

    cout << "前序遍历结果: ";
    preorderTraversal(root); // 输出: 1 2 4 5 3 
    cout << endl;

    // 释放内存 (实际应用中应使用智能指针或析构函数)
    return 0;
}

```

### Map映射
map 是一种有序的键值对容器，底层实现是红黑树。与 unordered_map 不同，它保证键的顺序，查找、插入和删除的时间复杂度为 O(log n)。

特点：
- 保证元素按键的顺序排列。
- 使用二叉搜索树实现。
  
优缺点：
- 优点：元素有序，适合需要按顺序处理数据的场景。
- 缺点：操作效率比 unordered_map 略低。

```cpp
map<string, int> myMap;
myMap["apple"] = 10;
cout << myMap["apple"]; // 输出 10
```

### Set集合
set 是一种用于存储唯一元素的有序集合，底层同样使用红黑树实现。它保证元素不重复且有序。

特点：
- 保证元素的唯一性。
- 元素自动按升序排列。
- 时间复杂度为 O(log n)。

优缺点：
- 优点：自动排序和唯一性保证。
- 缺点：插入和删除的效率不如无序集合。

```cpp
set<int> s;
s.insert(1);
s.insert(2);
cout << *s.begin(); // 输出 1
```

### Union联合体
Union是一种特殊的数据结构，允许在同一内存位置存储不同类型的数据，但同一时间只能使用其中的一种类型。联合体的大小取决于其最大成员的大小。

### Enum枚举
枚举是一种用户定义的类型，包含一组命名的常量。枚举使代码更具可读性和可维护性。

```cpp  
enum Color { RED, GREEN, BLUE };
```

## 存储类

存储类定义 C++ 程序中变量/函数的范围（可见性）和生命周期。这些说明符放置在它们所修饰的类型之前。下面列出 C++ 程序中可用的存储类：
-   **auto**：这是默认的存储类说明符，通常可以省略不写。auto 指定的变量具有自动存储期，即它们的生命周期仅限于定义它们的块（block）。auto 变量通常在栈上分配。
-   **register**：用于建议编译器将变量存储在CPU寄存器中以提高访问速度。在 C++11 及以后的版本中，register 已经是一个废弃的特性，不再具有实际作用。
-   **static**：用于定义具有静态存储期的变量或函数，它们的生命周期贯穿整个程序的运行期。在函数内部，static变量的值在函数调用之间保持不变。在文件内部或全局作用域，static变量具有内部链接，只能在定义它们的文件中访问。
-   **extern**：用于声明具有外部链接的变量或函数，它们可以在多个文件之间共享。默认情况下，全局变量和函数具有 extern 存储类。在一个文件中使用extern声明另一个文件中定义的全局变量或函数，可以实现跨文件共享。
-   **mutable (C++11)**：用于修饰类中的成员变量，允许在const成员函数中修改这些变量的值。通常用于缓存或计数器等需要在const上下文中修改的数据。
-   **thread_local (C++11)**：用于定义具有线程局部存储期的变量，每个线程都有自己的独立副本。线程局部变量的生命周期与线程的生命周期相同。

从 C++11 开始，register 已经失去了原有的作用，而 mutable 和 thread_local 则是新引入的特性，用于解决特定的编程问题。

示例：
```cpp
#include <iostream>

// 全局变量，具有外部链接，默认存储类为extern
int globalVar;
void function() {
    // 局部变量，具有自动存储期，默认存储类为auto
    auto int localVar = 10;
    // 静态变量，具有静态存储期，生命周期贯穿整个程序
    static int staticVar = 20;
    const int constVar = 30; // const变量默认具有static存储期
    // 尝试修改const变量，编译错误
    // constVar = 40;
    // mutable成员变量，可以在const成员函数中修改
    class MyClass {
    public:
        mutable int mutableVar;
        void constMemberFunc() const {
            mutableVar = 50; // 允许修改mutable成员变量
        }
    };
    // 线程局部变量，每个线程有自己的独立副本
    thread_local int threadVar = 60;
}

int main() {
    extern int externalVar; // 声明具有外部链接的变量
    function();
    return 0;
}
```

### auto
自 C++ 11 以来，auto 关键字用于两种情况：声明变量时根据初始化表达式自动推断该变量的类型、声明函数时函数返回值的占位符。C++98 标准中 auto 关键字用于自动变量的声明，但由于使用极少且多余，在 C++17 中已删除这一用法。

### register
register 是一种存储类（storage class），用于声明变量，并提示编译器将这些变量存储在寄存器中，以便快速访问。使用 register 关键字可以提高程序的执行速度，因为它减少了对内存的访问次数。

然而，需要注意的是，register 存储类只是一种提示，编译器可以忽略它，因为现代的编译器通常会自动优化代码，选择合适的存储位置，因此使用register 关键字并不是必需的，而且在实践中很少使用。

在 C++11 标准中，register 关键字不再是一个存储类说明符，而是一个废弃的特性。这意味着在 C++11 及以后的版本中，使用 register 关键字将不会对程序产生任何影响。在 C++ 中，可以使用引用或指针来提高访问速度，尤其是在处理大型数据结构时。

### static
static 存储类指示编译器在程序的生命周期内保持局部变量的存在，而不需要在每次它进入和离开作用域时进行创建和销毁。因此，使用 static 修饰局部变量可以在函数调用之间保持局部变量的值。

static 修饰符也可以应用于全局变量。当 static 修饰全局变量时，会使变量的作用域限制在声明它的文件内。在 C++ 中，当 static 用在类数据成员上时，会导致仅有一个该成员的副本被类的所有对象共享。

### extern
extern 存储类用于提供一个全局变量的引用，全局变量对所有的程序文件都是可见的。当您使用 'extern' 时，对于无法初始化的变量，会把变量名指向一个之前定义过的存储位置。

当您有多个文件且定义了一个可以在其他文件中使用的全局变量或函数时，可以在其他文件中使用 extern 来得到已定义的变量或函数的引用。可以这么理解，extern 是用来在另一个文件中声明一个全局变量或函数。

extern 修饰符通常用于当有两个或多个文件共享相同的全局变量或函数的时候，如下所示：
```cpp
// main.cpp
#include <iostream>
int count ;
extern void write_extern();
int main()
{
   count = 5;
   write_extern();
}
// support.cpp
#include <iostream>
extern int count;
void write_extern(void)
{
   std::cout << "Count is " << count << std::endl;
}
```

### mutable
mutable 是一个关键字，用于修饰类的成员变量，使其能够在 const 成员函数中被修改。通常情况下，const 成员函数不能修改对象的状态，但如果某个成员变量被声明为 mutable，则可以在 const 函数中对其进行修改。特点：
- 允许修改：mutable 成员变量可以在 const 成员函数内被改变。
- 设计目的：通常用于需要在不改变对象外部状态的情况下进行状态管理的场景，比如缓存、延迟计算等。

示例：
```cpp
#include <iostream>
class Example {
public:
    Example() : value(0), cachedValue(0) {}
    // 常量成员函数
    int getValue() const {
        return value; // 读取常量成员
    }
    // 修改 mutable 成员
    void increment() {
        ++value;
        cachedValue = value * 2; // 修改 mutable 成员
    }
    int getCachedValue() const {
        return cachedValue; // 读取 mutable 成员
    }
private:
    int value;                 // 常规成员，不能在 const 函数中修改
    mutable int cachedValue;   // 可修改成员，可以在 const 函数中修改
};
int main() {
    const Example ex;
    // ex.increment(); // 错误：无法在 const 对象上调用非 const 函数
    // ex.value = 10;  // 错误：无法修改 const 对象的成员
    std::cout << "Value: " << ex.getValue() << std::endl;
    std::cout << "Cached Value: " << ex.getCachedValue() << std::endl; // 输出为 0
    return 0;
}
```
适用场景：

- 缓存：在 const 函数中计算并缓存结果，而不影响对象的外部状态。
- 状态跟踪：如日志计数器，跟踪调用次数等信息，避免对类的逻辑进行侵入式修改。

注意事项：mutable 适用于需要在 const 环境中更改状态的特定情况，而不是普遍的设计模式，使用应谨慎，以免导致意外的状态变化，影响代码的可读性和可维护性。

### thread_local
thread_local 是 C++11 引入的一种存储类，用于在多线程环境中管理线程特有的变量。

使用 thread_local 修饰的变量在每个线程中都有独立的实例，因此每个线程对该变量的操作不会影响其他线程。

- 独立性：每个线程都有自己独立的变量副本，不同线程之间的读写操作互不干扰。
- 生命周期：thread_local 变量在其线程结束时自动销毁。
- 初始化：thread_local 变量可以进行静态初始化或动态初始化，支持在声明时初始化。

thread_local 适合用于需要存储线程状态、缓存或者避免数据竞争的场景，如线程池、请求上下文等。示例：

```cpp
#include <iostream>
#include <thread>
 
thread_local int threadSpecificVar = 0; // 每个线程都有自己的 threadSpecificVar
 
void threadFunction(int id) {
    threadSpecificVar = id; // 设置线程特有的变量
    std::cout << "Thread " << id << ": threadSpecificVar = " << threadSpecificVar << std::endl;
}
int main() {
    std::thread t1(threadFunction, 1);
    std::thread t2(threadFunction, 2);
    t1.join();
    t2.join();
    return 0;
}
```
注意事项：

- 性能：由于每个线程都有独立的副本，thread_local 变量的访问速度可能比全局或静态变量稍慢。
- 静态存储：thread_local 变量的存储类型为静态存储持续时间，因此在程序整个运行期间会一直存在。

## 引用

引用（Reference）是 C++ 中的一种类型，它为一个变量提供了一个别名。引用必须在定义时进行初始化，并且一旦绑定到一个变量，就不能再绑定到另一个变量。引用的语法是在类型后面加上一个 & 符号。

### 引用vs指针

|**特性**|**引用**|**指针**|
|---|---|---|
|**定义与初始化**|必须初始化，且不能为 `null`。|可以不初始化，可以在后续代码中指向其他对象，可以为 `null`。|
|**语法**|使用 `&` 声明，例如：`int &ref = a;`|使用 `*` 声明，例如：`int *ptr = &a;`|
|**重新绑定**|不能重新绑定，一旦初始化后始终引用同一个对象。|可以重新指向其他对象，例如：`ptr = &b;`|
|**内存占用**|不占用额外内存（编译器通常将其优化为直接操作所引用的对象）。|占用额外内存（存储地址，通常是一个机器字长，如4字节或8字节）。|
|**访问方式**|直接使用，无需解引用操作符，例如：`ref = 10;`|需要使用 `*` 解引用操作符访问或修改所指向的对象，例如：`*ptr = 10;`|
|**多级间接访问**|不支持多级间接访问（不能有引用的引用）。|支持多级间接访问（如指针的指针：`int **pptr;`）。|
|**函数参数传递**|常用于函数参数传递，语法简洁，例如：`void func(int &x) { x = 10; }`|也可以用于函数参数传递，但需要使用解引用操作符，例如：`void func(int *x) { *x = 10; }`|
|**数组与引用**|不能直接创建引用数组，但可以创建数组的引用，例如：`int (&ref)[10] = arr;`|可以创建指针数组，也可以创建指向数组的指针，例如：`int *ptrArr[10];`|
|**安全性**|更安全，不能为 `null`，且语法更直观。|更灵活，但容易出错（如空指针、野指针等）。|
|**底层实现**|通常通过指针实现，但编译器会优化为直接操作所引用的对象。|直接存储目标对象的内存地址。|

创建引用示例：
```cpp
#include <iostream>
using namespace std;
int main() {
    int a = 10;
    int &ref = a; // 创建一个引用，ref 是 a 的别名
    std::cout << "a: " << a << ", ref: " << ref << std::endl; // 输出a和ref的值
    ref = 20; // 修改 ref 的值，也会修改 a 的值
    std::cout << "a: " << a << ", ref: " << ref << std::endl; // 输出修改后的a和ref的值
    return 0;
}
```

### 把引用作为函数参数
示例：
```cpp
#include <iostream>
using namespace std;

void swap(int& x, int& y);
int main ()
{
   int a = 100;
   int b = 200;
   cout << "交换前，a 的值：" << a << endl;
   cout << "交换前，b 的值：" << b << endl;
   swap(a, b);
   cout << "交换后，a 的值：" << a << endl;
   cout << "交换后，b 的值：" << b << endl;
   return 0;
}

void swap(int& x, int& y)
{
   int temp;
   temp = x; /* 保存地址 x 的值 */
   x = y;    /* 把 y 赋值给 x */
   y = temp; /* 把 x 赋值给 y  */
  
   return;
}
```
### 把引用作为返回值
示例：
```cpp
#include <iostream>
using namespace std;
 
double vals[] = {10.1, 12.6, 33.1, 24.1, 50.0};

double& setValues(int i) {  
   double& ref = vals[i];    
   return ref;   // 返回第 i 个元素的引用，ref 是一个引用变量，ref 引用 vals[i]
}
int main ()
{
   cout << "改变前的值" << endl;
   for ( int i = 0; i < 5; i++ ){
       cout << "vals[" << i << "] = ";
       cout << vals[i] << endl;
   }
 
   setValues(1) = 20.23; // 改变第 2 个元素
   setValues(3) = 70.8;  // 改变第 4 个元素
   cout << "改变后的值" << endl;
   for ( int i = 0; i < 5; i++ ){
       cout << "vals[" << i << "] = ";
       cout << vals[i] << endl;
   }
   return 0;
}

## 智能指针

### weak_ptr
weak_ptr 是 C++11 引入的一种智能指针，用于解决 shared_ptr 循环引用的问题。weak_ptr 不拥有所指向的对象，因此不会增加对象的引用计数。它提供了一种安全的方式来访问 shared_ptr 管理的对象，而不会导致内存泄漏。
```cpp

#include <iostream>
#include <memory>

struct Person {
    std::string name;
    std::shared_ptr<Person> friend_shared; // 强引用
    std::weak_ptr<Person> friend_weak;     // 弱引用

    Person(std::string n) : name(n) { std::cout << name << " 出生了\n"; }
    ~Person() { std::cout << name << " 离开了\n"; }
};

int main() {
    auto xiao_ming = std::make_shared<Person>("小明");
    auto xiao_hong = std::make_shared<Person>("小红");

    // 小明强引用小红
    xiao_ming->friend_shared = xiao_hong;
    // 小红弱引用小明（关键！）
    xiao_hong->friend_weak = xiao_ming;

    std::cout << "当前小明的引用计数: " << xiao_ming.use_count() << "\n"; // 依然是 1
    
    return 0; // 函数结束，xiao_ming 和 xiao_hong 都能正常析构
}
```

## 面向对象

### 类与对象
C++ 在 C 语言的基础上增加了面向对象编程，C++ 支持面向对象程序设计。类是 C++ 的核心特性，通常被称为用户定义的类型。类用于指定对象的形式，是一种用户自定义的数据类型，它是一种封装了数据和函数的组合。类中的数据称为成员变量，函数称为成员函数。类可以被看作是一种模板，可以用来创建具有相同属性和行为的多个对象。

#### 类的定义

```cpp
class ClassName {
    AccessSpecifier:            //访问修饰符：public、private、protected
    DataType memberVariable;    //成员变量
    MemberFunction();           //成员函数 
};
```

#### 定义对象
类是对象的蓝图或模板，而对象是类的实例。通过类定义对象，可以创建具有相同属性和行为的多个对象。

```cpp
Class Book
{
    public:
    string title;
    string author;
    int year;
}; 

Book book1; // 创建一个 Book 类的对象 book1
Book book2; // 创建另一个 Book 类的对象 book2
```

#### 访问数据成员
使用点运算符（.）访问对象的成员变量和成员函数，但private和protected成员只能在类的内部访问，不能在类的外部访问。

#### 成员函数的定义方法
可以定义在内部也可以在外部使用::运算符来定义成员函数。

```cpp
class Book
{
    public:
    string title;
    void setTitle(string t) { // 内部定义成员函数
        title = t;
    }
    void printTitle(); // 声明成员函数
};
void Book::printTitle() { // 外部定义成员函数
    cout << "Title: " << title << endl;
}
```
#### 类访问修饰符

|修饰符|现实类比|谁可以访问？|典型用途|
|---|---|---|---|
|**public**|**客厅/大门**|**所有人**（类内部、子类、外部代码）|对外提供的接口函数（API）。|
|**protected**|**卧室**|**你和你的孩子**（类内部、子类）|仅限家族内部（继承体系）使用的数据。|
|**private**|**保险箱**|**只有你自己**（仅限本类内部）|核心数据、不想被随意修改的变量。|

如果不写访问修饰符，默认是 private。

##### public
公有成员在程序中任何地方都能访问，无需通过成员函数读写。

##### private
私有成员对类外完全封闭，外部代码无法读取、修改或调用，派生类同样无权直接访问。只有类自身的成员函数与被授予友元权限的实体能够操作这些内容。
```cpp
#include <iostream>
using namespace std;
class Box
{
   public:
      double length;
      void setWidth( double wid );
      double getWidth( void );
 
   private:
      double width;
};
double Box::getWidth(void)
{
    return width ;
}
void Box::setWidth( double wid )
{
    width = wid;
}
int main( )
{
   Box box;
   box.length = 10.0; 
   cout << "Length of box : " << box.length <<endl;

   // box.width = 10.0; // Error: 因为 width 是私有的
   box.setWidth(10.0);  // 使用成员函数设置宽度
   cout << "Width of box : " << box.getWidth() <<endl;
   return 0;
}
```
##### protected
protected 的存在主要是为了继承。如果没有继承，它和 private 一样（外人不可见）。如果有继承，子类（派生类）可以访问父类的 protected 成员，但不能访问 private 成员。
```cpp
#include <iostream>
using namespace std;
class Box
{
   protected:
      double width;
};
 
class SmallBox:Box // SmallBox 是派生类
{
   public:
      void setSmallWidth( double wid );
      double getSmallWidth( void );
};
 
// 子类的成员函数
double SmallBox::getSmallWidth(void)
{
    return width ;
}
 
void SmallBox::setSmallWidth( double wid )
{
    width = wid;
}
 
// 程序的主函数
int main( )
{
   SmallBox box;
   // 使用成员函数设置宽度
   box.setSmallWidth(5.0);
   cout << "Width of box : "<< box.getSmallWidth() << endl;
   return 0;
}
```

### 类的构造函数与析构函数

#### 构造函数
类的构造函数是类的一种特殊的成员函数，它会在每次创建类的新对象时执行。构造函数的名称与类的名称是完全相同的，并且不会返回任何类型，也不会返回 void。构造函数可用于为某些成员变量设置初始值。

示例：
```cpp
#include <iostream>
using namespace std;
class Line
{
   public:
      void setLength( double len );
      double getLength( void );
      Line();  // 这是构造函数
   private:
      double length;
};
// 成员函数定义，包括构造函数
Line::Line(void)
{
    cout << "Object is being created" << endl;
}
void Line::setLength( double len )
{
    length = len;
}
double Line::getLength( void )
{
    return length;
}
// 程序的主函数
int main( )
{
   Line line;
   // 设置长度
   line.setLength(6.0); 
   cout << "Length of line : " << line.getLength() <<endl;
 
   return 0;
}
```
或者直接使用带参数的构造函数：
```cpp
class Line
{
   public:
      void setLength( double len );
      double getLength( void );
      Line(double len);  // 这是构造函数
 
   private:
      double length;
};
Line::Line(double len)
{
    length = len;
}
// 等同于使用初始化列表：
//Line::Line( double len): length(len)
//{
    //cout << "Object is being created, length = " << len << endl;
//}
```
#### 析构函数
类的析构函数是类的一种特殊的成员函数，它会在每次删除所创建的对象时执行。析构函数的名称与类的名称是完全相同的，只是在前面加了个波浪号（~）作为前缀，它不会返回任何值，也不能带有任何参数。析构函数有助于在跳出程序（比如关闭文件、释放内存等）前释放资源。

示例：
```cpp
#include <iostream>
using namespace std;
class Line{
   public:
      void setLength( double len );
      double getLength( void );
      Line();   // 这是构造函数声明
      ~Line();  // 这是析构函数声明
   private:
      double length;
};
Line::Line(void){
    cout << "Object is being created" << endl;
}
Line::~Line(void){
    cout << "Object is being deleted" << endl;
}
 
void Line::setLength( double len ){
    length = len;
}
 
double Line::getLength( void ){
    return length;
}

int main( ){
   Line line;
   line.setLength(6.0); 
   cout << "Length of line : " << line.getLength() <<endl;
   return 0;
}
```
### 拷贝构造函数
拷贝构造函数是一种特殊的构造函数，它在创建对象时，是使用同一类中之前创建的对象来初始化新创建的对象。拷贝构造函数通常用于：
- 通过使用另一个同类型的对象来初始化新创建的对象。
- 复制对象把它作为参数传递给函数。
- 复制对象，并从函数返回这个对象。

如果类带有指针变量，并由动态分配内存，则它必须有一个拷贝构造函数

示例：
```cpp
#include <iostream>
using namespace std;
 
class Line{
   public:
      int getLength( void );
      Line( int len );             // 简单的构造函数
      Line( const Line &obj);      // 拷贝构造函数
      ~Line();                     // 析构函数
   private:
      int *ptr;
};

// 成员函数定义，包括构造函数
Line::Line(int len){
    cout << "调用构造函数" << endl;
    // 为指针分配内存
    ptr = new int;
    *ptr = len;
}
Line::Line(const Line &obj){
    cout << "调用拷贝构造函数并为指针 ptr 分配内存" << endl;
    ptr = new int;
    *ptr = *obj.ptr; // 拷贝值
}
Line::~Line(void){
    cout << "释放内存" << endl;
    delete ptr;
}

int Line::getLength( void )
{
    return *ptr;
}
 
void display(Line obj)
{
   cout << "line 大小 : " << obj.getLength() <<endl;
}
 
// 程序的主函数
int main( )
{
   Line line(10);
   display(line); // 传递对象 line 给函数 display，触发拷贝构造函数，obj 是 line 的一个副本
   return 0;
}
```

#### 拷贝构造函数详解

**普通构造函数**：在堆上申请内存，存入值。但当类中有**指针**时，问题来了：
```cpp
// 如果没有拷贝构造函数，默认的复制行为是"浅拷贝"
Line line1(10);
Line line2 = line1;  // 只复制了指针的地址，两者指向同一块内存！
```
此时 line1 和 line2 的 ptr 都指向同一块内存，修改 line1 的值会影响 line2，反之亦然。这就是浅拷贝的问题。同时，当其中一个对象被销毁时，另一个对象的 ptr 仍然指向已释放的内存，导致悬空指针问题。


**拷贝构造函数**：重新分配内存，再复制值（**深拷贝**）。通过`ptr = new int;`为每个对象分配独立的内存空间，确保 line1 和 line2 各自拥有自己的数据副本，互不干扰。

### 友元函数
类的友元函数是定义在类外部，但有权访问类的所有私有（private）成员和保护（protected）成员。尽管友元函数的原型有在类的定义中出现过，但是友元函数并不是成员函数。友元可以是一个函数，该函数被称为友元函数；友元也可以是一个类，该类被称为友元类，在这种情况下，整个类及其所有成员都是友元。

如果要声明函数为一个类的友元，需要在类定义中该函数原型前使用关键字 friend，如下所示：
```cpp
#include <iostream>
 
using namespace std;
 
class Box{
   double width;
public:
   friend void printWidth( Box box );
   void setWidth( double wid );
};
 
// 成员函数定义
void Box::setWidth( double wid ){
    width = wid;
}
 
// 请注意：printWidth() 不是任何类的成员函数
void printWidth( Box box ){
   /* 因为 printWidth() 是 Box 的友元，它可以直接访问该类的任何成员 */
   cout << "Width of box : " << box.width <<endl;
}
 
// 程序的主函数
int main( ){
   Box box;
   // 使用成员函数设置宽度
   box.setWidth(10.0);
   // 使用友元函数输出宽度
   printWidth( box );
   return 0;
}
```

### this指针
在 C++ 中，this 指针是一个特殊的指针，它指向当前对象的实例，每一个对象都能通过 this 指针来访问自己的地址。this是一个隐藏的指针，可以在类的成员函数中使用，它可以用来指向调用对象。当一个对象的成员函数被调用时，编译器会隐式地传递该对象的地址作为 this 指针。友元函数没有 this 指针，因为友元不是类的成员，只有成员函数才有 this 指针。

示例：
```cpp
#include <iostream>
 
class MyClass {
private:
    int value;
public:
    void setValue(int value) { // 传入参数名与成员变量名相同
        this->value = value;
    }
    void printValue() {
        std::cout << "Value: " << this->value << std::endl;
    }
};
int main() {
    MyClass obj;
    obj.setValue(42);
    obj.printValue();
    return 0;
}
```

### 指向类的指针
一个指向 C++ 类的指针与指向结构的指针类似，访问指向类的指针的成员，需要使用成员访问运算符 ->，就像访问指向结构的指针一样。与所有的指针一样，必须在使用指针之前，对指针进行初始化。
```cpp
#include <iostream>

class MyClass {
public:
    int data;
    void display() {
        std::cout << "Data: " << data << std::endl;
    }
};
int main() {
    MyClass obj;
    obj.data = 42;
    // 声明和初始化指向类的指针
    MyClass *ptr = &obj;
    // 通过指针访问成员变量
    std::cout << "Data via pointer: " << ptr->data << std::endl;
    // 通过指针调用成员函数
    ptr->display();
    return 0;
}
```
指向类的指针可以用于动态分配内存，创建类的对象:`MyClass *ptr = new MyClass;`；也可以作为函数参数传递：`void processObject(MyClass *ptr) {ptr->display();}`。

### 类的静态成员
可以使用static关键字将类的成员定义为静态的。当我们声明类的成员为静态时，**这意味着无论创建多少个类的对象，静态成员都只有一个副本**。

静态成员在类的所有对象中是共享的。如果不存在其他的初始化语句，在创建第一个对象时，所有的静态数据都会被初始化为零。我们不能把静态成员的初始化放置在类的定义中，但是可以在类的外部通过使用范围解析运算符 `::` 来重新声明静态变量从而对它进行初始化。

#### 静态成员函数
如果把函数成员声明为静态的，就可以把函数与类的任何特定对象独立开来。静态成员函数即使在类对象不存在的情况下也能被调用，静态函数只要使用类名加范围解析运算符 `::` 就可以访问。

静态成员函数只能访问静态成员数据、其他静态成员函数和类外部的其他函数。静态成员函数有一个类范围，他们不能访问类的 this 指针，可以使用静态成员函数来判断类的某些对象是否已被创建。

静态成员函数与普通成员函数的区别：

- 静态成员函数没有 this 指针，只能访问静态成员（包括静态成员变量和静态成员函数）。
- 普通成员函数有 this 指针，可以访问类中的任意成员；而静态成员函数没有 this 指针。

## 继承
面向对象程序设计中最重要的一个概念是继承。继承允许我们依据另一个类来定义一个类，这使得创建和维护一个应用程序变得更容易。这样做，也达到了重用代码功能和提高执行效率的效果。

当创建一个类时，不需要重新编写新的数据成员和成员函数，只需指定新建的类继承了一个已有的类的成员即可。这个已有的类称为基类，新建的类称为派生类，也被称为父类和子类。派生类可以添加新的成员变量和成员函数，也可以重写基类的成员函数来实现不同的行为。

```cpp
// 基类
class Shape{};
// 派生类
class Rectangle: public Shape{};
```
### 继承中的访问权限
|访问|public|protected|private|
|---|---|---|---|
|同一个类|yes|yes|yes|
|派生类|yes|yes|no|
|外部的类|yes|no|no|

一个派生类继承了所有的基类方法，但下列情况除外：

- 基类的构造函数、析构函数和拷贝构造函数。
- 基类的重载运算符。
- 基类的友元函数。
### 继承类型
当一个类派生自基类，该基类可以被继承为 **public、protected** 或 **private** 几种类型。继承类型是通过上面讲解的访问修饰符 access-specifier 来指定的。

我们几乎不使用 **protected** 或 **private** 继承，通常使用 **public** 继承。当使用不同类型的继承时，遵循以下几个规则：

-   **公有继承（public）：**当一个类派生自**公有**基类时，基类的**公有**成员也是派生类的**公有**成员，基类的**保护**成员也是派生类的**保护**成员，基类的**私有**成员不能直接被派生类访问，但是可以通过调用基类的**公有**和**保护**成员来访问。
-   **保护继承（protected）：** 当一个类派生自**保护**基类时，基类的**公有**和**保护**成员将成为派生类的**保护**成员。
-   **私有继承（private）：**当一个类派生自**私有**基类时，基类的**公有**和**保护**成员将成为派生类的**私有**成员。

### 多继承
一个类可以派生自多个类，这意味着，它可以从多个基类继承数据和函数。定义一个派生类，我们使用一个类派生列表来指定基类。通过以如下形式进行定义：
```cpp
class Derived: access-specifier Base1, access-specifier Base2,..., access-specifier BaseN
{
   // 派生类的成员
};
```

## 重载运算符和重载函数
C++ 允许在同一作用域中的某个**函数**和**运算符**指定多个定义，分别称为**函数重载**和**运算符重载**。

重载声明是指一个与之前已经在该作用域内声明过的函数或方法具有相同名称的声明，但是它们的参数列表和定义（实现）不相同。

当调用一个**重载函数**或**重载运算符**时，编译器通过把所使用的参数类型与定义中的参数类型进行比较，决定选用最合适的定义。选择最合适的重载函数或重载运算符的过程，称为**重载决策**。
### 函数重载
函数重载示例：
```cpp
#include <iostream>
using namespace std;
 
class printData
{
   public:
      void print(int i) {
        cout << "整数为: " << i << endl;
      }
      void print(double  f) {
        cout << "浮点数为: " << f << endl;
      }
      void print(char c[]) {
        cout << "字符串为: " << c << endl;
      }
};
int main(void)
{
   printData pd;
   pd.print(5);
   pd.print(500.263);
   char c[] = "Hello C++";
   pd.print(c);
   return 0;
}
```
### 运算符重载
重载的运算符是带有特殊名称的函数，函数名是由关键字 operator 和其后要重载的运算符符号构成的。与其他函数一样，重载运算符有一个返回类型和一个参数列表，格式如下：

运算符重载示例：
```cpp
#include <iostream>
using namespace std;
 
class Box
{
   public:
      double getVolume(void){
         return length * breadth * height;
      }
      void setLength( double len ){
          length = len;
      }
      void setBreadth( double bre ){
          breadth = bre;
      }
      void setHeight( double hei ){
          height = hei;
      }
      // 重载 + 运算符，用于把两个 Box 对象相加
      Box operator+(const Box& b)
      {
         Box box;
         box.length = this->length + b.length;
         box.breadth = this->breadth + b.breadth;
         box.height = this->height + b.height;
         return box;
      }
   private:
      double length;      // 长度
      double breadth;     // 宽度
      double height;      // 高度
};
// 程序的主函数
int main( )
{
   Box Box1;                // 声明 Box1，类型为 Box
   Box Box2;                // 声明 Box2，类型为 Box
   Box Box3;                // 声明 Box3，类型为 Box
   double volume = 0.0;     // 把体积存储在该变量中
 
   // Box1 详述
   Box1.setLength(6.0); 
   Box1.setBreadth(7.0); 
   Box1.setHeight(5.0);
 
   // Box2 详述
   Box2.setLength(12.0); 
   Box2.setBreadth(13.0); 
   Box2.setHeight(10.0);
   // Box1 的体积
   volume1 = Box1.getVolume();
   cout << "Volume of Box1 : " << volume1 <<endl;
   // Box2 的体积
   volume2 = Box2.getVolume();
   cout << "Volume of Box2 : " << volume2 <<endl;
   // 把两个对象相加，得到 Box3
   Box3 = Box1 + Box2;
   // Box3 的体积
   volume3 = Box3.getVolume();
   cout << "Volume of Box3 : " << volume3 <<endl;
   if(volume1 + volume2 != volume3)
       cout <<"Volumes do not match!" << endl;
   return 0;
}
```

## 多态

在 C++ 中，多态（Polymorphism）是面向对象编程的重要特性之一。多态按字面的意思就是多种形态。当类之间存在层次结构，并且类之间是通过继承关联时，就会用到多态。

C++ 多态允许使用基类指针或引用来调用子类的重写方法，从而使得同一接口可以表现不同的行为多态使得代码更加灵活和通用，程序可以通过基类指针或引用来操作不同类型的对象，而不需要显式区分对象类型。这样可以使代码更具扩展性，在增加新的形状类时不需要修改主程序。

### 关键点
#### 虚函数（Virtual Functions）
在基类中声明一个函数为虚函数，使用关键字virtual。派生类可以重写（override）这个虚函数。
调用虚函数时，会根据对象的实际类型来决定调用哪个版本的函数。

#### 动态绑定（Dynamic Binding）
也称为晚期绑定（Late Binding），在运行时确定函数调用的具体实现。需要使用指向基类的指针或引用来调用虚函数，编译器在运行时根据对象的实际类型来决定调用哪个函数。

#### 纯虚函数（Pure Virtual Functions）
一个包含纯虚函数的类被称为抽象类（Abstract Class），它不能被直接实例化。纯虚函数没有函数体，声明时使用= 0。

纯虚函数表示基类定义了一个接口，但具体实现由派生类负责（必须重写）。纯虚函数使得基类变为抽象类（abstract class），无法实例化。

|特性|虚函数（Virtual Function）|纯虚函数（Pure Virtual Function）|
|---|---|---|
|定义|基类中使用 `virtual` 声明，有实现|基类中使用 `= 0` 声明，无实现|
|子类重写|子类可以选择重写|子类必须实现|
|抽象性|可以实例化类|使类变为抽象类，无法实例化|
|用途|提供默认行为，允许子类重写|定义接口，强制子类实现具体行为|

### 多态的实现机制
使用虚函数表（V-Table）：C++运行时使用虚函数表来实现多态。每个包含虚函数的类都有一个虚函数表，表中存储了指向类中所有虚函数的指针。

使用虚指针（V-Pointer）：每个对象包含一个指向其类的虚函数表的指针。当通过基类指针调用虚函数时，程序通过虚指针找到正确的函数实现。

从示例来理解多态的概念：

示例1：
```cpp
#include <iostream>
using namespace std;

class Animal {
public:
    // 虚函数 sound，为不同的动物发声提供接口
    virtual void sound() const {
        cout << "Animal makes a sound" << endl;
    }
    
    virtual ~Animal() { 
        cout << "Animal destroyed" << endl; 
    }
};

class Dog : public Animal {
public:
    void sound() const override {
        cout << "Dog barks" << endl;
    }
    ~Dog() {
        cout << "Dog destroyed" << endl;
    }
};

class Cat : public Animal {
public:
    void sound() const override {
        cout << "Cat meows" << endl;
    }
    ~Cat() {
        cout << "Cat destroyed" << endl;
    }
};

// 测试多态
int main() {
    Animal* animalPtr;  // 定义基类指针
    // 创建 Dog 对象，并指向 Animal 指针
    animalPtr = new Dog();
    animalPtr->sound();  // 调用 Dog 的 sound 方法
    delete animalPtr;    // 释放内存，调用 Dog 和 Animal 的析构函数
    // 创建 Cat 对象，并指向 Animal 指针
    animalPtr = new Cat();
    animalPtr->sound();  // 调用 Cat 的 sound 方法
    delete animalPtr;    // 释放内存，调用 Cat 和 Animal 的析构函数
    return 0;
}
```
示例2：
```cpp
#include <iostream>
using namespace std;

class Shape {
   protected:
      int width, height; // 宽度和高度
   public:
      // 构造函数，带有默认参数
      Shape(int a = 0, int b = 0) : width(a), height(b) { }
      virtual int area() {
         cout << "Shape class area: " << endl;
         return 0;
      }
};
 
class Rectangle : public Shape {
   public:
      Rectangle(int a = 0, int b = 0) : Shape(a, b) { }
      int area() override { 
         cout << "Rectangle class area: " << endl;
         return width * height;
      }
};
 
class Triangle : public Shape {
   public:
      Triangle(int a = 0, int b = 0) : Shape(a, b) { }
      int area() override { 
         cout << "Triangle class area: " << endl;
         return (width * height / 2); 
      }
};
 
// 主函数
int main() {
   Shape *shape;           // 基类指针
   Rectangle rec(10, 7);   // 矩形对象
   Triangle tri(10, 5);    // 三角形对象
   shape = &rec;
   cout << "Rectangle Area: " << shape->area() << endl;
   shape = &tri;
   cout << "Triangle Area: " << shape->area() << endl;
 
   return 0;
}
```

### 多态的优势
- 代码复用：通过基类指针或引用，可以操作不同类型的派生类对象，实现代码的复用。
- 扩展性：新增派生类时，不需要修改依赖于基类的代码，只需要确保新类正确重写了虚函数。
- 解耦：多态允许程序设计更加模块化，降低类之间的耦合度。

## 数据抽象
数据抽象是指，只向外界提供关键信息，并隐藏其后台的实现细节，即只表现必要的信息而不呈现细节。数据抽象是一种依赖于接口和实现分离的编程（设计）技术。

C++ 类为数据抽象提供了可能。它们向外界提供了大量用于操作对象数据的公共方法，也就是说，外界实际上并不清楚类的内部实现。

### 访问标签
在 C++ 中，我们使用访问标签来定义类的抽象接口，即public、private、protected三个关键字。一个类可以包含零个或多个访问标签。

使用公共标签定义的成员都可以访问该程序的所有部分。一个类型的数据抽象视图是由它的公共成员来定义的。

使用私有标签定义的成员无法访问到使用类的代码。私有部分对使用类型的代码隐藏了实现细节。

访问标签出现的频率没有限制。每个访问标签指定了紧随其后的成员定义的访问级别。指定的访问级别会一直有效，直到遇到下一个访问标签或者遇到类主体的关闭右括号为止。

### 数据抽象的好处
- 类的内部受到保护，不会因无意的用户级错误导致对象状态受损。
- 类实现可能随着时间的推移而发生变化，以便应对不断变化的需求，或者应对那些要求不改变用户级代码的错误报告。

### 接口
接口描述了类的行为和功能，而不需要完成类的特定实现。

C++ 接口是使用抽象类来实现的，抽象类与数据抽象互不混淆，数据抽象是一个把实现细节与相关的数据分离开的概念。如果类中至少有一个函数被声明为纯虚函数，则这个类就是抽象类。纯虚函数是通过在声明中使用 "= 0" 来指定的。

设计抽象类（通常称为 ABC）的目的，是为了给其他类提供一个可以继承的适当的基类。抽象类不能被用于实例化对象，它只能作为接口使用。如果试图实例化一个抽象类的对象，会导致编译错误。

## 数据封装
数据封装（Data Encapsulation）是面向对象编程（OOP）的一个基本概念，它通过将数据和操作数据的函数封装在一个类中来实现。这种封装确保了数据的私有性和完整性，防止了外部代码对其直接访问和修改。数据封装是一种把数据和操作数据的函数捆绑在一起的机制，数据抽象是一种仅向用户暴露接口而把具体的实现细节隐藏起来的机制。

所有的 C++ 程序都有以下两个基本要素：

- 程序语句（代码）：这是程序中执行动作的部分，它们被称为函数。
- 程序数据：数据是程序的信息，会受到程序函数的影响。

封装是面向对象编程中的把数据和操作数据的函数绑定在一起的一个概念，这样能避免受到外界的干扰和误用，从而确保了安全。数据封装引申出了另一个重要的 OOP 概念，即数据隐藏。

C++ 通过创建类来支持封装和数据隐藏（public、protected、private）。我们已经知道，类包含私有成员（private）、保护成员（protected）和公有成员（public）成员。

## 命名空间
在C++中，可能会出现一个库中的函数与另一个库中的函数同名的情况。为了避免这种命名冲突，C++引入了命名空间的概念。命名空间是一种封装标识符的机制，可以将标识符分组到不同的命名空间中，从而避免命名冲突。本质上，命名空间就是定义了一个范围。
```cpp
#include <iostream>
using namespace std;

namespace first_space {
    void myFunction() {
        std::cout << "Hello from first_space!" << std::endl;
    }
}
namespace second_space {
    void myFunction() {
        std::cout << "Hello from second_space!" << std::endl;
    }
}
int main() {
    first_space::myFunction(); // 调用 first_space 中的函数
    second_space::myFunction(); // 调用 second_space 中的函数
    return 0;
}
```
### using指令
使用using指令可以将命名空间中的标识符引入当前作用域，从而可以直接使用这些标识符而不需要加上命名空间的前缀。
```cpp
using namespace std;
cout << "Hello, World!" << endl; // 直接使用cout，而endl还是需要std:: 前缀
```
需要注意的是，使用using指令可能会引入命名冲突，特别是在大型项目中。因此，建议在使用using指令时要谨慎，避免引入不必要的命名冲突。

using 指令也可以用来指定命名空间中的特定项目，例如：
```cpp
using std::cout;
cout << "Hello, World!" << std::endl; // 直接使用 cout，无需 std:: 前缀
```
### 不连续的命名空间
C++ 允许在不同的地方定义同一个命名空间，这被称为不连续的命名空间。这意味着可以在一个文件中定义命名空间的一部分，在另一个文件中定义同一命名空间的另一部分。这种特性使得代码组织更加灵活，可以将相关的功能分散在不同的文件中，同时仍然保持在同一个命名空间下。
```cpp
// file1.cpp
#include <iostream>
namespace my_namespace {
    void function1() {
        std::cout << "This is function1 in my_namespace." << std::endl;
    }
}       
// file2.cpp
#include <iostream>
namespace my_namespace {
    void function2() {
        std::cout << "This is function2 in my_namespace." << std::endl;
    }
}
// main.cpp
#include <iostream>
namespace my_namespace {
    void function1();
    void function2();
}
int main() {
    my_namespace::function1();
    my_namespace::function2();
    return 0;
}
```
在这个例子中，`my_namespace` 的定义被分散在 `file1.cpp` 和 `file2.cpp` 中，而在 `main.cpp` 中，我们可以直接使用 `my_namespace` 中的函数，而不需要担心它们的定义位置。这种不连续的命名空间使得代码的组织更加灵活，可以根据需要将相关的功能分散在不同的文件中，同时保持在同一个命名空间下。

### 嵌套的命名空间
C++ 还支持嵌套的命名空间，这意味着一个命名空间可以包含另一个命名空间。这种特性使得代码的组织更加层次化，可以更好地管理大型项目中的代码。
```cpp
#include <iostream>
namespace outer {
    void function1() {
        std::cout << "This is function1 in outer namespace." << std::endl;
    }
    namespace inner {
        void function2() {
            std::cout << "This is function2 in inner namespace." << std::endl;
        }
    }
}
int main() {
    outer::function1(); // 调用 outer 命名空间中的函数
    outer::inner::function2(); // 调用 inner 命名空间中的函数
    return 0;
}
```
在这个例子中，`outer` 命名空间包含了一个名为 `inner` 的嵌套命名空间。我们可以通过 `outer::function1()` 来调用 `outer` 命名空间中的函数，而通过 `outer::inner::function2()` 来调用 `inner` 命名空间中的函数。这种嵌套的命名空间使得代码的组织更加层次化，可以更好地管理大型项目中的代码。

## 文件和流
标准库<fstream>提供了文件输入输出的功能，允许我们在 C++ 程序中读写文件。文件流（fstream）是一个类，用于处理文件的输入输出操作。它提供了三个主要的类：
- `ifstream`：用于从文件中读取数据的输入文件流。
- `ofstream`：用于向文件中写入数据的输出文件流。
- `fstream`：既可以用于读取也可以用于写入的文件流。

### 打开文件
使用`open()`成员函数打开文件，它是上述三个类的成员函数：
```cpp
void open(const char* filename, ios::openmode mode);
```
openmode如下：

|模式标志|描述|
|---|---|
|ios::app|追加模式。所有写入都追加到文件末尾。|
|ios::ate|文件打开后定位到文件末尾。|
|ios::in|打开文件用于读取。|
|ios::out|打开文件用于写入。|
|ios::trunc|如果该文件已经存在，其内容将在打开文件之前被截断，即把文件长度设为 0。|

以写入模式打开文件，并希望截断文件，以防文件已存在：
```cpp
ofstream outFile;
outFile.open("file.dat", ios::out | ios::trunc);
```
打开文件用于读写：
```cpp
fstream file;
file.open("file.dat", ios::out | ios::in);
```

### 关闭文件
```cpp
void close();
```
### 写入和读取文件
类似<iostream>，使用流插入运算符（ << ）向文件写入信息，使用流提取运算符（ >> ）从文件读取信息。

### 文件位置指针

<istream> 和 <ostream> 都提供了用于重新定位文件位置指针的成员函数。这些成员函数包括关于 <istream> 的 `seekg`和关于 <ostream> 的 `seekp`。

`seekg` 和 `seekp` 的参数通常是一个长整型。第二个参数可以用于指定查找方向。查找方向可以是 ios::beg（默认的，从流的开头开始定位），也可以是 ios::cur（从流的当前位置开始定位），也可以是 ios::end（从流的末尾开始定位）。

文件位置指针是一个整数值，指定了从文件的起始位置到指针所在位置的字节数。下面是关于定位 "get" 文件位置指针的实例：
```cpp
// 定位到 fileObject 的第 n 个字节（假设是 ios::beg）
fileObject.seekg( n );
// 把文件的读指针从 fileObject 当前位置向后移 n 个字节
fileObject.seekg( n, ios::cur );
// 把文件的读指针从 fileObject 末尾往回移 n 个字节
fileObject.seekg( n, ios::end );
// 定位到 fileObject 的末尾
fileObject.seekg( 0, ios::end );
```

## 异常处理
### 抛出异常
`throw` 语句用于抛出一个异常。当程序执行到 `throw` 语句时，程序会停止当前函数的执行，并将控制权转移到最近的异常处理程序（catch 块）。`throw` 语句可以抛出任何类型的对象，通常是一个错误消息或者一个错误代码。
```cpp
double division(int a, int b)
{
   if( b == 0 )
   {
      throw "Division by zero condition!";
   }
   return (a/b);
}
```
### 捕获异常
catch 块跟在 try 块后面，用于捕获异常。
```cpp
try
{
   // 保护代码
}catch( ExceptionName e ) //catch(...)用于catch 块能够处理 try 块抛出的任何类型的异常
{
  // 处理 ExceptionName 异常的代码
}
```
示例：
```cpp
#include <iostream>
using namespace std;
double division(int a, int b){
   if( b == 0 )
   {
      throw "Division by zero condition!";
   }
   return (a/b);
}
 
int main (){
   int x = 50;
   int y = 0;
   double z = 0;
   try {
     z = division(x, y);
     cout << z << endl;
   }catch (const char* msg) {
     cerr << msg << endl;
   }
   return 0;
}
```
### 常见的异常
|异常|描述|
|---|---|
|**std::exception**|该异常是所有标准 C++ 异常的父类。|
|std::bad\_alloc|该异常可以通过 **new** 抛出。|
|std::bad\_cast|该异常可以通过 **dynamic\_cast** 抛出。|
|std::bad\_typeid|该异常可以通过 **typeid** 抛出。|
|std::bad\_exception|这在处理 C++ 程序中无法预期的异常时非常有用。|
|**std::logic\_error**|理论上可以通过读取代码来检测到的异常。|
|std::domain\_error|当使用了一个无效的数学域时，会抛出该异常。|
|std::invalid\_argument|当使用了无效的参数时，会抛出该异常。|
|std::length\_error|当创建了太长的 std::string 时，会抛出该异常。|
|std::out\_of\_range|该异常可以通过方法抛出，例如 std::vector 和 std::bitset<>::operator\[\]()。|
|**std::runtime\_error**|理论上不可以通过读取代码来检测到的异常。|
|std::overflow\_error|当发生数学上溢时，会抛出该异常。|
|std::range\_error|当尝试存储超出范围的值时，会抛出该异常。|
|std::underflow\_error|当发生数学下溢时，会抛出该异常。|

### 利用继承和重载定义新的异常
```cpp
#include <iostream>
#include <exception>
using namespace std;
 
struct MyException : public exception
{
  const char * what () const throw ()
  {
    return "C++ Exception";
  }
};
 
int main()
{
  try
  {
    throw MyException();
  }
  catch(MyException& e)
  {
    std::cout << "MyException caught" << std::endl;
    std::cout << e.what() << std::endl;
  }
  catch(std::exception& e)
  {
    //其他的错误
  }
}
```

## 动态内存
C++ 程序中的内存分为两个部分：
- 栈：在函数内部声明的所有变量都将占用栈内存。
- 堆：这是程序中未使用的内存，在程序运行时可用于动态分配内存。

### new和delete运算符
C++ 提供了 `new` 和 `delete` 运算符来动态分配。使用`new`运算符为给定类型的变量在运行时分配堆内的内存，这会返回所分配的空间地址，使用`delete`运算符来释放之前分配的内存。
```cpp
new data-type;
```

在这里，**data-type** 可以是包括数组在内的任意内置的数据类型，也可以是包括类或结构在内的用户自定义的任何数据类型。让我们先来看下内置的数据类型。例如，我们可以定义一个指向 double 类型的指针，然后请求内存，该内存在执行时被分配。我们可以按照下面的语句使用 **new** 运算符来完成这点：
```cpp
double* pvalue = NULL; 
pvalue = new double;
```
如果自由存储区已被用完，可能无法成功分配内存。所以建议检查 new 运算符是否返回 NULL 指针，并采取以下适当的操作：
```cpp
double* pvalue = NULL; 
if( !(pvalue = new double )) { 
    cout << "Error: out of memory." <<endl; 
    exit(1); 
}
```
**malloc()** 函数在 C 语言中就出现了，在 C++ 中仍然存在，但建议尽量不要使用 malloc() 函数。new 与 malloc() 函数相比，其主要的优点是，new 不只是分配了内存，它还创建了对象。

使用`delete`运算符来释放之前分配的内存：
```cpp
delete pvalue;
```
示例：
```cpp
#include <iostream>
using namespace std;
 
int main (){
   double* pvalue  = NULL; // 初始化为 null 的指针
   pvalue  = new double;   // 为变量请求内存
   *pvalue = 29494.99;     // 在分配的地址存储值
   cout << "Value of pvalue : " << *pvalue << endl;
   delete pvalue;         // 释放内存
   return 0;
}
```
### 数组的动态内存分配
一维数组：
```cpp
int *array = new int[10];
delete [] array;
```
二维数组：
```cpp
int **array;
array = new int*[m];
for (int i = 0; i < m; i++){
    array[i] = new int[n];
}
for (int i = 0; i < m; i++){
    delete [] array[i];
}
delete [] array;
```
三维数组：
```cpp
int ***array;
array = new int **[m];
for( int i=0; i<m; i++ )
{
    array[i] = new int *[n];
    for( int j=0; j<n; j++ )
    {
        array[i][j] = new int [h];
    }
}
for( int i=0; i<m; i++ )
{
    for( int j=0; j<n; j++ )
    {
        delete[] array[i][j];
    }
    delete[] array[i];
}
delete[] array;
```
### 对象的动态内存分配
示例：
```cpp
#include <iostream>
using namespace std;
 
class Box{
   public:
      Box() { 
         cout << "调用构造函数！" <<endl; 
      }
      ~Box() { 
         cout << "调用析构函数！" <<endl; 
      }
};
 
int main( ){
   Box* myBoxArray = new Box[4];
   delete [] myBoxArray; // 删除数组
   return 0;
}
```

## 模板
模板是泛型编程的基础，泛型编程即以一种独立于任何特定类型的方式编写代码。模板是创建泛型类或函数的蓝图或公式。库容器，比如迭代器和算法，都是泛型编程的例子，它们都使用了模板的概念。

每个容器都有一个单一的定义，比如 **向量**，我们可以定义许多不同类型的向量，比如 **vector \<int>** 或 **vector \<string>**。

### 函数模板
```cpp
template <typename type> ret-type func-name(parameter list)
{
   // 函数的主体
}
```
示例：
```cpp
#include <iostream>
#include <string>
using namespace std;

template <typename T>
inline T const& Max (T const& a, T const& b) { 
    return a < b ? b:a; 
} 
int main (){
    int i = 39;
    int j = 20;
    cout << "Max(i, j): " << Max(i, j) << endl; 
    double f1 = 13.5; 
    double f2 = 20.7; 
    cout << "Max(f1, f2): " << Max(f1, f2) << endl; 
    string s1 = "Hello"; 
    string s2 = "World"; 
    cout << "Max(s1, s2): " << Max(s1, s2) << endl; 
    return 0;
}
```
### 类模板
```cpp
泛型类声明的一般形式如下所示：
template <class type> class class-name {
...
}
```
type 是占位符类型名称，可以在类被实例化的时候进行指定。您可以使用一个逗号分隔的列表来定义多个泛型数据类型。

下面的实例定义了类 Stack<>，并实现了泛型方法来对元素进行入栈出栈操作：
```cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <stdexcept>
 
using namespace std;
 
template <class T>
class Stack { 
  private: 
    vector<T> elems;            // 用 vector 作为底层存储，元素类型是 T
 
  public: 
    void push(T const&);        // 入栈
    void pop();                 // 出栈
    T top() const;              // 返回栈顶元素
    bool empty() const{         // 如果为空则返回真。
        return elems.empty(); 
    } 
};                              // 类定义结束
// 类外定义成员函数
template <class T>
void Stack<T>::push (T const& elem) { 
    // 追加传入元素的副本
    elems.push_back(elem);    
} 
 
template <class T>
void Stack<T>::pop () 
{ 
    if (elems.empty()) { 
        throw out_of_range("Stack<>::pop(): empty stack"); 
    }
    // 删除最后一个元素
    elems.pop_back();         
} 
 
template <class T>
T Stack<T>::top () const 
{ 
    if (elems.empty()) { 
        throw out_of_range("Stack<>::top(): empty stack"); 
    }
    // 返回最后一个元素的副本 
    return elems.back();      
} 
 
int main() 
{ 
    try { 
        Stack<int>         intStack;  // int 类型的栈 
        Stack<string> stringStack;    // string 类型的栈 
 
        // 操作 int 类型的栈 
        intStack.push(7); 
        cout << intStack.top() <<endl; 
 
        // 操作 string 类型的栈 
        stringStack.push("hello"); 
        cout << stringStack.top() << std::endl; 
        stringStack.pop(); 
        stringStack.pop(); 
    } 
    catch (exception const& ex) { 
        cerr << "Exception: " << ex.what() <<endl; 
        return -1;
    } 
}
```
返回：
> 7
> 
> hello
> 
> Exception: pop(): empty stack

## 预处理
预处理器是一些指令，指示编译器在实际编译之前所需完成的预处理。所有的预处理器指令都是以井号（#）开头，只有空格字符可以出现在预处理指令之前。预处理指令不是 C++ 语句，所以它们不会以分号（;）结尾。

我们已经看到，之前所有的实例中都有 #include 指令。这个宏用于把头文件包含到源文件中。

### #define指令宏
#define 预处理指令用于创建符号常量。如`#define PI 3.14159`。当使用`-E`进行编译时，会看到预处理器的输出，其中所有的 PI 都被替换为 3.14159。
```bash
$ gcc -E test.cpp > test.p

...
int main ()
{
 
    cout << "Value of PI :" << 3.14159 << endl; 

    return 0;
}
```

### 参数宏
可以使用`#define`来指定一个带参数的宏，如`#define MIN(a,b) (a<b ? a : b)`。

### 条件编译

有几个指令可以用来有选择地对部分程序源代码进行编译。这个过程被称为条件编译。条件预处理器的结构与 if 选择结构很像。请看下面这段预处理器的代码：
```cpp
#ifdef NULL
   #define NULL 0
#endif
```
可以只在调试时进行编译，调试开关可以使用一个宏来实现，如下所示：
```cpp
#ifdef DEBUG
   cerr <<"Variable x = " << x << endl;
#endif //如果在指令 #ifdef DEBUG 之前已经定义了符号常量 DEBUG，则会对程序中的 cerr 语句进行编译。
```
使用`#if 0`注释不需要编译的代码：
```cpp
#if 0
    这段代码不会被编译
#endif
```
### #和##运算符
#运算符会把 replacement-text 令牌转换为用引号引起来的字符串。
```cpp
#include <iostream>
using namespace std;
#define MKSTR( x ) #x
int main (){
    cout << MKSTR(HELLO C++) << endl; //输出"HELLO C++"
    return 0;
}
```
##运算符会把两个令牌连接成一个令牌。
```cpp
#include <iostream>
using namespace std;
 
#define concat(a, b) a ## b
int main(){
   int xy = 100;
   cout << concat(x, y); //输出100
   return 0;
}
```
### 预定义宏
|宏|描述|
|---|---|
|\_\_LINE\_\_|这会在程序编译时包含当前行号。|
|\_\_FILE\_\_|这会在程序编译时包含当前文件名。|
|\_\_DATE\_\_|这会包含一个形式为 month/day/year 的字符串，它表示把源文件转换为目标代码的日期。|
|\_\_TIME\_\_|这会包含一个形式为 hour:minute:second 的字符串，它表示程序被编译的时间。|

## 信号处理
信号是由操作系统传给进程的中断，会提早终止一个程序。在 UNIX、LINUX、Mac OS X 或 Windows 系统上，可以通过按 `Ctrl+C` 产生中断。

|信号|描述|
|---|---|
|SIGABRT|程序的异常终止，如调用 **abort**。|
|SIGFPE|错误的算术运算，比如除以零或导致溢出的操作。|
|SIGILL|检测非法指令（编号9）。|
|SIGINT|程序终止(interrupt)信号（编号2）。|
|SIGSEGV|非法访问内存（编号11）。|
|SIGTERM|发送到程序的终止请求（编号15）。|

### signal()函数
一般形式为`signal(registered signal, signal handler)`，这个函数接收两个参数：第一个参数是要设置的信号的标识符，如SIGINT，第二个参数是指向信号处理函数的指针。函数返回值是一个指向先前信号处理函数的指针。如果先前没有设置信号处理函数，则返回值为 SIG_DFL。如果先前设置的信号处理函数为 SIG_IGN，则返回值为 SIG_IGN。

让我们编写一个简单的 C++ 程序，使用 signal() 函数捕获 SIGINT 信号。不管想在程序中捕获什么信号，都必须使用 signal 函数来注册信号，并将其与信号处理程序相关联。

示例：
```cpp
#include <iostream>
#include <csignal>
#include <unistd.h>
 
using namespace std;
 
void signalHandler( int signum ){
    cout << "Interrupt signal (" << signum << ") received.\n";
    exit(signum);  
 
}
int main (){
    signal(SIGINT, signalHandler);  
    while(1){
       cout << "Going to sleep...." << endl;
       sleep(1);
    }
    return 0;
}
```
按下ctrl+c，输出：
> Going to sleep....
>
> Going to sleep....
>
> Going to sleep....
>
> Interrupt signal (2) received. // 2 是 SIGINT 的信号编号

### raise()函数
`raise()` 函数用于向当前进程发送一个信号。它接受一个一个整数信号编号作为参数，语法如下：
```cpp
int raise(signal sig);
```
在这里，**sig** 是要发送的信号的编号，这些信号包括：SIGINT、SIGABRT、SIGFPE、SIGILL、SIGSEGV、SIGTERM、SIGHUP。

## 多线程
**线程**是程序中的轻量级执行单元，允许程序同时执行多个任务。多线程是多任务处理的一种特殊形式，多任务处理允许让电脑同时运行两个或两个以上的程序。

### 线程、进程
进程(Process)是一个正在运行的程序的实例，具有独立的地址空间、内存、数据栈以及其他用于跟踪进程执行的辅助数据。每个进程至少有一个线程，即主线程。

线程(Thread)是程序执行中的单一顺序控制流，多个线程可以在同一个进程中独立运行。线程共享进程的地址空间、文件描述符、堆和全局变量等资源，但每个线程有自己的栈、寄存器和程序计数器。

一般情况下，两种类型的多任务处理：基于进程和基于线程。
- 基于进程的多任务处理是程序的并发执行。
- 基于线程的多任务处理是同一程序的片段的并发执行。

### 并发与并行
并发(Concurrency)是指在同一时间段内，多个任务交替执行。并行(Parallelism)是指在同一时间点上，多个任务在多个处理器或核上同时执行。

### 创建线程
C++ 11 之后添加了新的标准线程库 std::thread，std::thread 在 <thread> 头文件中声明，因此使用 std::thread 时需要包含 在 <thread> 头文件。
```cpp
#include<thread>
std::thread thread_object(callable, args...);
```
其中，**callable** 是一个可调用对象，可以是一个函数指针、一个函数对象或者一个 lambda 表达式。**args...** 是传递给可调用对象的参数列表。

#### 使用函数指针
```cpp
#include <iostream>
#include <thread>

void printMessage(int count) {
    for (int i = 0; i < count; ++i) {
        std::cout << "Hello from thread (function pointer)!\n";
    }
}

int main() {
    std::thread t1(printMessage, 5);    // 创建线程，传递函数指针和参数
    t1.join();                          // 等待线程完成
    return 0;
}
```
使用`g++ -std=c++11`编译后输出：
>Hello from thread (function pointer)!
>
>Hello from thread (function pointer)!
>
>Hello from thread (function pointer)!
>
>Hello from thread (function pointer)!
>
>Hello from thread (function pointer)!  
>
>Hello from thread (function pointer)!

#### 使用函数对象
通过类中的 operator() 方法定义函数对象来创建线程：
```cpp
#include <iostream>
#include <thread>

class PrintTask {
public:
    void operator()(int count) const {  //重载 operator() 方法，使其成为一个可调用对象
        // const表明此函数为常量成员函数，不会修改对象的任何成员变量
        for (int i = 0; i < count; ++i) {
            std::cout << "Hello from thread (function object)!\n";
        }
    }
};

int main() {
    std::thread t2(PrintTask(), 5); // 创建线程，传递函数对象和参数
    t2.join(); // 等待线程完成
    return 0;
}    
```

#### 使用lambda表达式
Lambda 表达式可以直接内联定义线程执行的代码：
```cpp
#include <iostream>
#include <thread>

int main() {
    std::thread t3([](int count) {
        for (int i = 0; i < count; ++i) {
            std::cout << "Hello from thread (lambda)!\n";
        }
    }, 5); // 创建线程，传递 Lambda 表达式和参数
    t3.join(); // 等待线程完成
    return 0;
}
```

### 线程管理
`join()` 用于等待线程完成执行。

`detach()` 将线程与主线程分离，线程在后台独立运行，主线程不再等待它。

如果不调用 join() 或 detach() 而直接销毁线程对象，会导致程序崩溃。

### 线程的传参
#### 值传递
```cpp
std::thread t(func, arg1, arg2);
```
#### 引用传递
使用`std::ref`来传递：
```cpp
#include <iostream>
#include <thread>

void increment(int& x) {
    ++x;
}

int main() {
    int num = 0;
    std::thread t(increment, std::ref(num)); // 使用 std::ref 传递引用
    t.join();
    std::cout << "Value after increment: " << num << std::endl;
    return 0;
}
```

### 线程同步与互斥
在多线程编程中，线程同步与互斥是两个非常重要的概念，它们用于控制多个线程对共享资源的访问，以避免数据竞争、死锁等问题。

#### 1. 互斥量（Mutex）

互斥量是一种同步原语，用于防止多个线程同时访问共享资源。当一个线程需要访问共享资源时，它首先需要锁定（lock）互斥量。如果互斥量已经被其他线程锁定，那么请求锁定的线程将被阻塞，直到互斥量被解锁（unlock）。`std::mutex`：用于保护共享资源，防止数据竞争。
```cpp
std::mutex mtx;
mtx.lock();   // 锁定互斥锁
// 访问共享资源
mtx.unlock(); // 释放互斥锁
```
互斥量的使用示例：
```cpp
#include <mutex>
std::mutex mtx; // 全局互斥量

void safeFunction() {
    mtx.lock(); // 请求锁定互斥量
    // 访问或修改共享资源
    mtx.unlock(); // 释放互斥量

}
int main() {
    std::thread t1(safeFunction);
    std::thread t2(safeFunction);
    t1.join();
    t2.join();
    return 0;
}
```
#### 2. 锁（Locks）

C++提供了多种锁类型，用于简化互斥量的使用和管理。

常见的锁类型包括：
-  `std::lock_guard`：作用域锁，当构造时自动锁定互斥量，当析构时自动解锁。
-  `std::unique_lock`：与`std::lock_guard`类似，但提供了更多的灵活性，例如可以转移所有权和手动解锁。

锁的使用示例：
```cpp
#include <mutex>
std::mutex mtx;

void safeFunctionWithLockGuard() {
    std::lock_guard<std::mutex> lk(mtx); // 自动锁定互斥量
    // 访问或修改共享资源
}

void safeFunctionWithUniqueLock() {
    std::unique_lock<std::mutex> ul(mtx);
    // 访问或修改共享资源
    // ul.unlock(); // 可选：手动解锁
    // ...
}
```
#### 3. 条件变量（Condition Variable）

条件变量用于线程间的协调，允许一个或多个线程等待某个条件的发生。它通常与互斥量一起使用，以实现线程间的同步。`std::condition_variable`用于实现线程间的等待和通知机制。
```cpp
std::condition_variable cv;
std::mutex mtx;
bool ready = false;
std::unique_lock<std::mutex> lock(mtx);
cv.wait(lock, []{ return ready; }); // 等待条件满足
// 条件满足后执行
```

条件变量的使用示例：
```cpp
#include <mutex>
#include <condition\_variable>
std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void workerThread() {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, []{ return ready; }); // 等待条件
    // 当条件满足时执行工作
}

void mainThread() {
    {
        std::lock_guard<std::mutex> lk(mtx);
        // 准备数据
        ready = true;
    } // 离开作用域时解锁
    cv.notify_one(); // 通知一个等待的线程
}
```
#### 4. 原子操作（Atomic Operations）

原子操作确保对共享数据的访问是不可分割的，即在多线程环境下，原子操作要么完全执行，要么完全不执行，不会出现中间状态。

原子操作的使用示例：
```cpp
#include <atomic>
#include <thread>

std::atomic<int> count(0);

void increment() {
    count.fetch_add(1, std::memory_order_relaxed);
}
int main() {
    std::thread t1(increment);
    std::thread t2(increment);
    t1.join();
    t2.join();
    return count; // 应返回2
}
```
#### 5. 线程局部存储（Thread Local Storage, TLS）

线程局部存储允许每个线程拥有自己的数据副本。这可以通过`thread_local`关键字实现，避免了对共享资源的争用。

线程局部存储的使用示例：
```cpp
#include <iostream>
#include <thread>
thread_local int threadData = 0;
void threadFunction() {
    threadData = 42; // 每个线程都有自己的threadData副本
    std::cout << "Thread data: " << threadData << std::endl;
}
int main() {
    std::thread t1(threadFunction);
    std::thread t2(threadFunction);
    t1.join();
    t2.join();
    return 0;
}
```
#### 6. 死锁（Deadlock）和避免策略

死锁发生在多个线程互相等待对方释放资源，但没有一个线程能够继续执行。避免死锁的策略包括：
-   总是以相同的顺序请求资源。
-   使用超时来尝试获取资源。
-   使用死锁检测算法。

### 线程间通信
使用`std::future` 和 `std::promise`实现线程间的值传递。

示例：
```cpp
std::promise<int> p;
std::future<int> f = p.get_future();
std::thread t([&p] {
    p.set_value(10); // 设置值，触发 future
});
int result = f.get(); // 获取值
```

## STL

C++ 标准模板库（Standard Template Library，STL）是一套功能强大的 C++ 模板类和函数的集合，它提供了一系列通用的、可复用的算法和数据结构。

STL 的设计基于泛型编程，这意味着使用模板可以编写出独立于任何特定数据类型的代码。

STL 分为多个组件，包括容器（Containers）、迭代器（Iterators）、算法（Algorithms）、函数对象（Function Objects）和适配器（Adapters）等。

使用 STL 的好处:

- 代码复用：STL 提供了大量的通用数据结构和算法，可以减少重复编写代码的工作。
- 性能优化：STL 中的算法和数据结构都经过了优化，以提供最佳的性能。
- 泛型编程：使用模板，STL 支持泛型编程，使得算法和数据结构可以适用于任何数据类型。
- 易于维护：STL 的设计使得代码更加模块化，易于阅读和维护。

### 容器
容器是用来存储数据的序列，它们提供了不同的存储方式和访问模式。STL 中的容器可以分为三类：
- 1、序列容器：存储元素的序列，允许双向遍历。
  -   std::vector：动态数组，支持快速随机访问。
  -   std::deque：双端队列，支持快速插入和删除。
  -   std::list：链表，支持快速插入和删除，但不支持随机访问。

- 2、关联容器：存储键值对，每个元素都有一个键（key）和一个值（value），并且通过键来组织元素。
  -   std::set：集合，不允许重复元素。
  -   std::multiset：多重集合，允许多个元素具有相同的键。
  -   std::map：映射，每个键映射到一个值。
  -   std::multimap：多重映射，存储了键值对（pair），其中键是唯一的，但值可以重复，允许一个键映射到多个值。

- 3、无序容器（C++11 引入）：哈希表，支持快速的查找、插入和删除。
  -   std::unordered_set：无序集合。
  -   std::unordered_multiset：无序多重集合。
  -   std::unordered_map：无序映射。
  -   std::unordered_multimap：无序多重映射。

详细解释见数据结构一节。

## 常见的标准库
###  \<vector>
提供了向量容器的实现。向量是一个动态数组，可以在运行时动态调整大小。它提供了许多有用的成员函数，如`push_back()`、`size()`等。
\<vector> 是 STL 中的一个容器类，用于存储动态大小的数组。\<vector> 是一个序列容器，它允许用户在容器的末尾快速地添加或删除元素。与数组相比，\<vector> 提供了更多的功能，如自动调整大小、随机访问等。

在 C++ 中，需要包含头文件 \<vector>。以下是一些基本的语法：
1. 预分配容器的内存容量（capacity）:

`reserve()`函数用于预分配容器的内存容量（capacity），以防止在插入元素时进行多次重新分配。它不改变容器的大小（size）或元素数量，仅增加容量。使用reserve()可避免不必要的内存拷贝，提高性能，适用于已知元素数量的情况。
```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v;
    // 预留 100 个整数的空间
    v.reserve(100); 

    std::cout << "Size: " << v.size() << std::endl;         // 输出: 0
    std::cout << "Capacity: " << v.capacity() << std::endl; // 输出: >= 100

    return 0;
}
```
2. `size()`获取元素数量：
```cpp
 size_t size = myVector.size();
``` 
3. `ssize()`获取元素数量（返回值为 signed）：
```cpp
 ssize_t size = myVector.ssize();
```
4. 清空 `vector`：
```cpp
myVector.clear();
```
5. range

### \<cmath>
提供了数学函数的实现，如三角函数、指数函数、对数函数等。\<cmath> 中的函数通常接受 float 或 double 类型的参数，并返回相应类型的结果。对于 long double 类型，你可以使用 \<cmath> 中的函数，但需要在函数名后加上 l 后缀，例如 `sqrtl`。

### \<string>
提供了字符串类的实现。C++中的字符串类`std::string`比C语言中的字符数组更易于使用，并且提供了许多有用的成员函数，如`length()`、`substr()`等。

### \<algorithm>
C++ 标准库中的 \<algorithm> 头文件提供了一组用于操作容器（如数组、向量、列表等）的算法。这些算法包括排序、搜索、复制、比较等，它们是编写高效、可重用代码的重要工具。

\<algorithm> 头文件定义了一组模板函数，这些函数可以应用于任何类型的容器，只要容器支持迭代器。这些算法通常接受两个或更多的迭代器作为参数，表示操作的起始和结束位置。

大多数 `<algorithm>` 中的函数都遵循以下基本语法：
```cpp
algorithm_name(container.begin(), container.end(), ...);
```
这里的 `container` 是一个容器对象，`begin()` 和 `end()` 是容器的成员函数，返回指向容器开始和结束的迭代器。
#### 排序
对容器内的元素进行排序。
```cpp
sort(container.begin(), container.end(), compare_function);
```
`std::partial_sort`: 对部分区间排序，前 n 个元素为有序。
```cpp
std::partial_sort(vec.begin(), vec.begin() + n, vec.end());
```
`std::stable_sort`: 稳定排序，保留相等元素的相对顺序。
```cpp
std::stable_sort(vec.begin(), vec.end());
```
#### 搜索
在容器中查找与给定值匹配的第一个元素。
```cpp
auto it = find(container.begin(), container.end(), value);
```
如果找到，it 将指向匹配的元素；如果没有找到，it 将等于 container.end()。
示例（力扣hot100第一题）：
```cpp
//给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        std::unordered_map <int,int> map;
        for(int i = 0; i < nums.size(); i++){
            auto index = map.find(target - nums[i]);
            if(index != map.end()){ 了
                return {index->second, i}; 
            }
            map.insert(pair<int, int>(nums[i], i));
        }
        return{};
    }
};
```
`std::binary_search`: 对有序区间进行二分查找，`binary_search()` 函数在 [first, last) 区域内成功找到和 val 相等的元素，则返回 true；反之则返回 false。
```cpp
std::sort(vec.begin(), vec.end());  // 先排序
bool found = std::binary_search(vec.begin(), vec.end(), 4);
```
`std::find_if`: 查找第一个满足特定条件的元素。
```cpp
auto it = std::find_if(vec.begin(), vec.end(), [](int x) { return x > 3; });
```

#### 复制
将一个范围内的元素复制到另一个容器或数组。
```cpp
copy(source_begin, source_end, destination_begin);
```
示例：
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> source = {1, 2, 3, 4, 5};
    int destination[5];
    std::copy(source.begin(), source.end(), destination);
    for (int i = 0; i < 5; ++i) {
        std::cout << destination[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

#### 比较
比较两个容器或两个范围内的元素是否相等。
```cpp
bool result = equal(first1, last1, first2);
// 三个参数分别表示第一个范围的起始和结束迭代器，以及第二个范围的起始迭代器
```
或
```cpp
bool result = equal(first1, last1, first2, compare_function);
// 如果提供了 compare_function，则使用该函数进行比较，默认使用`==`进行比较。
```
注意，`std::equal` 函数只比较元素的值，而不关心序列中元素的排序或者它们在内存中的位置。因此两个序列必须具有相同的长度，否则不可能相等。
示例：
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {1, 2, 3, 4, 5};

    bool are_equal = std::equal(v1.begin(), v1.end(), v2.begin());
    std::cout << (are_equal ? "Vectors are equal." : "Vectors are not equal.") << std::endl;

    return 0;
}
```
```cpp
//自定义二元谓词
#include <iostream>
#include <vector>
#include <algorithm>
#include <cctype> // 为了 std::tolower

// 自定义二元谓词，忽略大小写比较两个字符串
struct ignore_case_equal {
    bool operator()(const std::string& str1, const std::string& str2) const {
        if (str1.size() != str2.size()) {
            return false;
        }
        for (size_t i = 0; i < str1.size(); ++i) {
            if (std::tolower(str1[i]) != std::tolower(str2[i])) {
                return false;
            }
        }
        return true;
    }
};

int main() {
    // 创建两个字符串向量
    std::vector<std::string> vec1 = {"Hello", "World", "!"};
    std::vector<std::string> vec2 = {"hello", "world", "!"};

    // 使用自定义的二元谓词比较两个向量是否相等（忽略大小写）
    bool areEqual = std::equal(vec1.begin(), vec1.end(), vec2.begin(), ignore_case_equal());
    if (areEqual) {
        std::cout << "The vectors are equal (ignoring case)." << std::endl;
    } else {
        std::cout << "The vectors are not equal." << std::endl;
    }
    return 0; //输出：The vectors are equal (ignoring case).
}
```

#### 修改
`std::reverse`: 反转区间内的元素顺序。
```cpp
std::reverse(vec.begin(), vec.end());
```
`std::fill`: 将指定区间内的所有元素赋值为某个值。
```cpp
std::fill(vec.begin(), vec.end(), 0);  // 所有元素设为 0
```
`std::replace`: 将区间内的某个值替换为另一个值。
```cpp
std::replace(vec.begin(), vec.end(), 1, 99);  // 将所有 1 替换为 99
```
`std::copy`: 将区间内的元素复制到另一个区间。
```cpp
std::vector<int> vec2(6);
std::copy(vec.begin(), vec.end(), vec2.begin());
```

#### 排列
`std::next_permutation`: 生成字典序的下一个排列，如果没有下一个排列则返回 false。
```cpp
std::vector<int> vec = {1, 2, 3};
do {
    for (int n : vec) std::cout << n << " ";
    std::cout << std::endl;
} while (std::next_permutation(vec.begin(), vec.end()));
```
`std::prev_permutation`: 生成字典序的上一个排列。
```cpp
std::prev_permutation(vec.begin(), vec.end());
```

示例：
```cpp
#include <iostream>
#include <algorithm>
using namespace std;
int main(){
    int num[3]={1,2,3};
    do{
        cout<<num[o]<<""<<num[1]<<""<<num[2]<<endl;
    }while(next_permutation(num,num+3)); //输出123的6种排列组合
    do{
        cout<<num[o]<<""<<num[1]<<""<<num[2]<<endl;
    }while(prev_permutation(num,num+2)); //输出123  213
    return 0;
}
```
另外，需要强调的是，`next_permutation()`在使用前需要对欲排列数组按升序排序，否则只能找出该序列之后的全排列数。比如，如果数组num初始化为[2,3,1]，那么输出就变为了:
> 231
>
> 312
>
> 321

#### 归并
`std::merge`: 将两个有序区间合并到一个有序区间。
```cpp
std::vector<int> vec1 = {1, 3, 5};
std::vector<int> vec2 = {2, 4, 6};
std::vector<int> result(6);
std::merge(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), result.begin());
```
`std::inplace_merge`: 在单个区间中合并两个有序子区间。
```cpp
std::inplace_merge(vec.begin(), middle, vec.end());// 合并[first, middle)和[middle, last)两个有序区间
```

#### 集合
`std::set_union`: 计算两个有序集合的并集。
```cpp
std::vector<int> result(10);
auto it = std::set_union(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), result.begin());
result.resize(it - result.begin());
```
`std::set_intersection`: 计算两个有序集合的交集。
```cpp
auto it = std::set_intersection(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), result.begin());
result.resize(it - result.begin());
```
`std::set_difference`: 计算集合的差集。
```cpp
auto it = std::set_difference(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), result.begin());
result.resize(it - result.begin());
```

## C++新特性

### C++11

#### lambda表达式
Lambda 表达式是一种匿名函数，可以直接在需要函数对象的地方定义和使用。它的语法如下：
```cpp
[capture_list] (parameter_list) -> return_type { function_body }
```
其中：
- capture_list：指定 lambda 表达式可以访问哪些外部变量，可以按值捕获（[=]）、按引用捕获（[&]）或全捕获（[]）。
- parameter_list：与普通函数参数相同。
- return_type：可选，通常可以由编译器自动推导。
- function_body：包含 lambda 表达式的具体实现。

示例：
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
int main() {
    std::vector<int> v = {3, 1, 4, 1, 5};
    int factor = 10;

    // 1. 基本使用：排序
    std::sort(v.begin(), v.end(), [](int a, int b) {
        return a < b; // 升序排列
    });

    // 2. 捕获外部变量 (按值捕获: factor)
    std::for_each(v.begin(), v.end(), [factor](int n) {
        std::cout << n * factor << " "; // 输出每个元素乘以 factor 的结果
    });

    return 0;
}
```

### C++14

### C++17

### C++20/C++23
从 C++20 开始，C++ 引入了模块（Modules），并在 C++23 中进一步完善了对标准库模块的支持。模块提供了一种更高效、更安全的方式来导入标准库。
-   **编译速度更快**：模块只编译一次，后续导入时直接使用编译好的二进制接口。
-   **隔离性更好**：模块不会泄露宏定义和私有符号，避免了命名冲突。
-   **依赖关系清晰**：模块的导入和导出机制使得代码的依赖关系更加清晰。
示例：
```cpp
import std; // 导入整个标准库（C++23 特性）
int main() {
    std::cout << "Hello, C++23 Modules!\\n"; // 使用 std::cout 输出
    return 0;
}
```
或者导入标准库的特定部分：
```cpp
import std.core;      // 导入核心库
import std.iostream;  // 导入输入输出流库
int main() {
    std::cout << "Hello, C++23 Modules!\\n";
    return 0;
}
```
目前，主流编译器对 C++ 模块的支持正在逐步完善。以下是一些编译器的支持情况和使用方法：
#### gcc
需要启用 C++23 标准和模块支持。

编译命令示例：
```bash
g++ -std=c++23 -fmodules-ts -o program main.cpp
```
#### clang
需要启用 C++23 标准和模块支持。

编译命令示例：
```bash
clang++ -std=c++23 -fmodules -o program main.cpp
```
#### MSVC（Visual Studio）
需要启用 C++23 标准和模块支持。

编译命令示例：

```bash
cl /std:c++23 /experimental:module /EHsc /Fe:program main.cpp
```

## 函数打桩（Function Stubbing）

函数打桩（Stubbing）是在测试或开发阶段，用一个临时替换的函数（桩函数）来代替真实函数的技术。目的是**隔离被测代码，排除外部依赖的干扰**。

### 驱动开发中的典型场景

驱动代码常常依赖：
- **硬件寄存器** — 读/写特定内存地址，但硬件还没就绪
- **内核 API** — 如 `kmalloc()`、`ioremap()` 等
- **其他模块的函数** — 而这些模块还没开发完成

打桩让你绕过这些依赖，先验证自己的逻辑，不必等所有依赖就绪。

### 编译期打桩（链接时替换）

利用链接器特性，让桩函数优先于原函数被链接：

```cpp
// 真实函数（在另一个 .o 或库中）
uint32_t read_register(uint32_t addr) {
    return *(volatile uint32_t *)addr;
}

// 桩函数 — 放在测试文件中，链接时会优先匹配
uint32_t read_register(uint32_t addr) {
    if (addr == 0x1000) return 0xDEADBEEF;
    return 0;
}
```

### 宏打桩（预处理期替换）

利用 `#define` 在编译前替换函数名：

```cpp
#ifdef ENABLE_STUB
#define read_register(addr)  read_register_stub(addr)
#endif

uint32_t read_register_stub(uint32_t addr) {
    return (addr == 0x1000) ? 0xDEADBEEF : 0;
}
```

### 函数指针打桩（运行期替换）

通过函数指针在运行时切换真实实现和桩实现，最灵活：

```cpp
// 定义一个函数指针类型
typedef uint32_t (*read_reg_fn)(uint32_t);

// 默认指向真实实现
read_reg_fn g_read_register = real_read_register;

// 测试时切换到桩
void stub_enable() {
    g_read_register = stub_read_register;
}

// 业务代码通过函数指针调用
uint32_t val = g_read_register(0x1000);
```

### 适用场景对比

| 方式 | 优点 | 缺点 |
|---|---|---|
| 链接期替换 | 对业务代码无侵入 | 需要控制链接顺序 |
| 宏替换 | 简单直接 | 不影响原函数内部调用，调试困难 |
| 函数指针 | 运行时灵活切换 | 有间接调用开销，需修改调用方式 |

## MOCK 与 STUB 的区别

### Stub（桩）

**只关心返回值，不关心调用过程。** 写死返回值来支撑被测代码运行。

```cpp
// Stub：不管输入什么，总是返回"成功"
int send_packet_stub(uint8_t *data, size_t len) {
    return 0;  // 永远成功，不关心调没调用、调用几次
}
```

### Mock（模拟）

**不仅返回预定义的值，还记录并验证调用过程。** 关注"被测代码有没有按约定调用依赖方"——调用了多少次、传了什么参数、调用顺序对不对。

```cpp
// Mock：记录调用信息，用于后续断言
class MockPacketSender {
public:
    int call_count = 0;
    size_t last_len = 0;
    std::vector<uint8_t> captured_data;

    int send(const uint8_t *data, size_t len) {
        call_count++;
        last_len = len;
        captured_data.assign(data, data + len);
        return 0;
    }
};

// 测试中验证行为：
// assert(mock.call_count == 1);
// assert(mock.last_len == expected_len);
// assert(mock.captured_data[0] == expected_header);
```

### 核心区别

| | **Stub（桩）** | **Mock（模拟）** |
|---|---|---|
| **关注点** | 返回值（状态） | 调用过程（交互行为） |
| **验证什么** | "被测代码收到什么数据" | "被测代码有没有正确调用了依赖" |
| **典型问题** | 它返回固定的假数据 | 可断言调用次数、参数、顺序 |
| **复杂度** | 简单，几行代码 | 需要记录调用历史并做断言 |

> 一句话：**Stub 问"你给我什么"，Mock 问"你有没有按约定调我"。**

驱动开发中更常见的是 Stub（模拟硬件寄存器返回值）。Mock 更多在单元测试框架（如 Google Mock）中使用，用来验证模块间的调用协议。

## 广播（内核通知链）

驱动开发中的"广播"通常指 Linux 内核的**通知链（Notifier Chain / Notification Chain）**机制——一种一对多的事件分发机制。一个模块发送事件，所有已注册的模块都会收到通知并做出响应。

### 为什么需要通知链

内核各子系统相互独立，但需要响应系统级事件。比如：
- 电源管理模块广播"系统即将休眠" → 所有驱动保存状态、关闭设备
- 网络子系统广播"网卡状态变化" → 相关模块刷新路由或连接
- USB 子系统广播"新设备插入" → 对应的驱动初始化

通知链实现了**发布者与订阅者的解耦**：发事件的一方不需要知道谁会响应。

### 四种通知链类型

| 类型 | 回调上下文 | 是否可阻塞 | 适用场景 |
|---|---|---|---|
| `atomic_notifier_chain` | 原子上下文（不可睡眠） | 否 | 中断、定时器中触发的事件 |
| `blocking_notifier_chain` | 进程上下文 | 是 | 普通事件通知，可调用可能阻塞的函数 |
| `raw_notifier_chain` | 无限制 | 取决于调用者 | 需要完全自定义锁保护的场景 |
| `srcu_notifier_chain` | 进程上下文 | 是 | 大量订阅者时的可伸缩通知链 |

驱动开发中最常用的是 `blocking_notifier_chain`。

### 基本用法

#### （1）定义通知链（事件频道）

```c
#include <linux/notifier.h>

// 定义一个阻塞型通知链，本质上是 struct blocking_notifier_head
static BLOCKING_NOTIFIER_HEAD(power_state_chain);

// 定义事件码（自定义）
#define EVENT_POWER_SUSPEND   0x01
#define EVENT_POWER_RESUME    0x02
#define EVENT_POWER_SHUTDOWN  0x03
```

#### （2）订阅广播（注册回调）

```c
// 回调函数：收到通知后做什么
static int my_power_callback(struct notifier_block *nb,
                              unsigned long event, void *data) {
    switch (event) {
    case EVENT_POWER_SUSPEND:
        // 保存设备状态，准备休眠
        save_device_state();
        break;
    case EVENT_POWER_RESUME:
        // 恢复设备状态
        restore_device_state();
        break;
    default:
        break;
    }
    return NOTIFY_OK;  // 继续通知链上的下一个回调
}

// notifier_block 将回调和通知链绑定
static struct notifier_block my_power_nb = {
    .notifier_call = my_power_callback,
};

// 注册（订阅）
static int __init my_driver_init(void)
{
    blocking_notifier_chain_register(&power_state_chain, &my_power_nb);
    return 0;
}
```

#### （3）发送广播（触发通知）

```c
// 某个模块发出广播：系统要休眠了
void trigger_suspend(void)
{
    // 通知链上所有注册的回调都会被依次调用
    blocking_notifier_call_chain(&power_state_chain,
                                  EVENT_POWER_SUSPEND, NULL);
}
```

#### （4）取消订阅

```c
static void __exit my_driver_exit(void)
{
    blocking_notifier_chain_unregister(&power_state_chain, &my_power_nb);
}
```

### 回调返回值

| 返回值 | 含义 |
|---|---|
| `NOTIFY_OK` | 正常处理，继续通知下一个 |
| `NOTIFY_DONE` | 正常处理，停止通知后续回调 |
| `NOTIFY_BAD` | 出错，停止通知后续回调 |
| `NOTIFY_STOP` | 停止通知，不表示错误 |

### 内核中常见的通知链实例

| 通知链 | 作用 |
|---|---|
| `reboot_notifier_list` | 系统重启/关机通知 |
| `netdev_chain` | 网络设备状态变化（up/down/注册/注销） |
| `usb_notifier_list` | USB 设备插拔事件 |
| `fb_notifier_list` | 帧缓冲（显示）相关事件 |
| `cpufreq_transition_notifier_list` | CPU 频率切换通知 |

### 一个完整的驱动场景

```c
// 场景：LCD 背光驱动需要在系统休眠时关屏，唤醒时亮屏

// 1. 注册到内核已有的通知链
static int lcd_notifier_callback(struct notifier_block *nb,
                                  unsigned long event, void *data) {
    switch (event) {
    case PM_SUSPEND_PREPARE:
        turn_off_backlight();
        break;
    case PM_POST_SUSPEND:
        turn_on_backlight();
        break;
    }
    return NOTIFY_OK;
}

static struct notifier_block lcd_pm_nb = {
    .notifier_call = lcd_notifier_callback,
};

// 2. 驱动初始化时订阅
static int __init lcd_init(void) {
    register_pm_notifier(&lcd_pm_nb);   // 内核自带的电源管理通知链
    return 0;
}

// 3. 驱动卸载时取消订阅
static void __exit lcd_exit(void) {
    unregister_pm_notifier(&lcd_pm_nb);
}
```

### 广播 vs 单播 vs 组播

| | 单播 (Unicast) | 广播 (Broadcast) | 组播 (Multicast) |
|---|---|---|---|
| 接收方数量 | 1个 | 所有注册者 | 指定的一组 |
| 消息投递 | 点到点 | 点到所有 | 点到组 |
| 内核类比 | 直接函数调用 | 通知链 | 等待队列 / 特定事件 |
| 耦合度 | 高 | 低（发布者不知道订阅者） | 中 |
