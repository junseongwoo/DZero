class Animal:
  pass 

cat = Animal()
print(isinstance(cat, Animal))

class Dog(Animal):
  pass 

dog = Dog() 
print(isinstance(dog, Animal))
print(isinstance(dog, Dog))
