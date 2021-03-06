"""
基本性质
从定义出发：
    左子树都比根节点小
    右子树都比根节点大
    如果有重复元素，可以自行选择放到左子树还是右子树

从效果出发：
    中序遍历 in-order traversal 是升序序列

性质：
    如果一棵二叉树的中序遍历不是升序，那一定不是BST
    如果一棵二叉树的中序遍历是升序,也未必是BST
        当存在重复元素是，相同的数要么同时在左子树，要么同时在右子树，不能一边一个

"""