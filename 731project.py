# First-program
'''
Terminal

1. git init

2. git add .   ->전부다 copy 한다는 뜻.
or git add (file_name.py)

3. git status
=>방금 추가한 정보를 알려준다.

4. git commit -m "first commit"
=>history create!

5. git remote add origin https://github.com/gheis123/First-program.git

6. git remote -v

7. git push origin master 
'''
#ctrl+k+c(주석)

#입출력하기, 빠른 입출력하기, List Comperhension, 3항연산자
#name=input()
#age=int(input())
#a,b=map(int,input().split())
#print(name,age,a,b)

#import sys
#input=sys.stdin.readline
#for _ in range(5):
#    n=input()
#    print(n)


##List Comperhension, 3항연산자
#first=[i for i in range(5)]
#print(first)
#second=[[1 for _ in range(3)] for _ in range(2)]
#print(second)

##3항 연산자
#from random import *
#num_list=[]
#for _ in range(randint(1,10)):
#    num_list.append(randint(1,50))
#print(num_list)
#shuffle(num_list)
#print(num_list)

##judgement
#for num in num_list:
#    even_or_odd="짝" if num%2==0 else "홀"
#    print("{} {}".format(num,even_or_odd))
#judgement=[1 if num%2!=0 else 0 for num in num_list]
#print(judgement)

#array: 연속적인 할당. random access , O(1), 삽입과 삭제의 속도가 느리다.
#따라서, 삽입과 삭제는 O(N)


#Linked list:메모리 상에 띄엄띄엄 위치할 수 있다[메모리공간 활용도가 높다]
#삽입과 삭제가 array에 비해 O(1)의 복잡도를 갖는다.
#하지만, random access가 불가능한다는 단점을 가지고 있다.

#요세푸스 문제
#원을 따라 앉아있고 (7,3)으로 입력하게 되면 3칸씩 띄어서 사람을 출력하고
#출력 대상에서 제외한다
#ex) 7 3 -><3,6,2,7,5,1,4>
#delete->location update->list.append
#(N,K)=map(int,input().split())
#peo=[i for i in range(1,N+1)]
#pt=0
#ans=[]
#for _ in range(N):
#    pt+=K-1 #시작시에는 2를 꺼내야 함. 
#    #아래에서 pop을 했기 때문에 계속 K-1씩 더해 나가야한다.
#    pt%=len(peo)
#    ans.append(peo.pop(pt))
#print(f"<{','.join(map(str,ans))}>")

#stack: FILO(first in last out ==last in first out)

#Valid PS VPS problem
#VPS라면 YES, 아니라면 "NO"를 출력하도록
def VPS_test(string):
    stack=[]
    judgement="YES"
    for strr in string:
        if strr=="(":
            stack.append(strr)
        else:
            if len(stack)>0:
                stack.pop()
            else:
                judgement="NO"

    if len(stack)>0: #들여쓰기가 매우 중요하다.
        judgement="NO"
    return judgement #마지막에 최종 보고하는 곳.


VPS_list=['(())())','(((()())()','(()())((()))','((()()(()))(((())))()',
      '()()()()(()()())()','(()((())()(']
judgement_VPS=[]
for i in VPS_list:
    judgement_VPS.append(VPS_test(i))

print(judgement_VPS)

    

