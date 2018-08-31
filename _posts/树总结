一，二叉树的遍历总结

二叉树的遍历，分为前序遍历，中序遍历，后序遍历，层次遍历；每种遍历又有递归和非递归两种方式，递归方式代码简洁、运行时间较短，唯一的潜在问题是过多的递归调用可能会导致栈溢出；非递归方式虽然能够避免栈溢出的问题，但是付出的代价是代码量多，而且需要掌握好边界条件的细节，此外，非递归方式的运行时间较长。

	大部分求解二叉树的问题，其本质都是基于以上四种遍历的基础上的。问题的切入点在于，对树节点的相关处理是先处理根节点再往左右子树方向（对应着前序遍历），还是先左子树，处理根节点，再往右子树走（对应着中序遍历）...，哪种遍历方式对解决当前问题更有利，就选择哪种遍历方式。

1.1 前序遍历

	前序遍历，节点访问顺序是先访问根节点，再访问左节点，最后访问右节点

递归法

    public List<Integer> preOrder(TreeNode root){
      List<Integer> ans = new ArrayList<>();
      if(root == null) return ans;
      dfs(root, ans);
      return ans;
    }
    public void dfs(TreeNode root, List<Integer> res){
      if(root == null) return ; //递归终止条件
      res.add(root.val);
      dfs(root.left, res);
      dfs(root.right, res);
    }

非递归

非递归法需要借助额外的数据结构：栈；从根节点开始，往左进行，每访问一个节点的时候，都将这个节点入栈，若当前节点为null时，表面已经到了左子树的底端，此时再出栈一个节点，继续往右走，代码如下：

    public List<Integer> preOrder(TreeNode root){
      List<Integer> ans = new ArrayList<>();
      if(root == null) return ans;
      Stack<TreeNode> s = new Stack<>();
      TreeNode cur = root;
      while(cur != null || !s.isEmpty()){
        if(cur != null){
          ans.add(cur.val);//先处理根节点
          s.push(cur);
          cur = cur.left;//再处理左节点
        }else{
          //左子树已经处理完，此时需要出栈一个节点，这个节点为左子树的最后一个节点
          TreeNode node = s.pop();
          cur = cur.right; //继续处理右节点
        }
      }
      return ans;
    }

1.2 关于前序遍历的例子

1.2.1 相同的树

给定两个二叉树，编写一个函数来检验它们是否相同（两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

    public boolean isSameTree(TreeNode p, TreeNode q){
      //先确定边界条件
      if(p == null && q != null) return false;
      if(p != null && q == null) return false;
      if(p == null && q == null) return true;
      
      if(p.val != q.val){
        return false;//处理根节点
      }
      boolean left = isSameTree(p.left, q.left);
      boolean right = isSameTree(p.right, q.right);
      return left && right;
    }

1.2.2 对称二叉树

给定一个二叉树，检查它是否是镜像对称。

此题与“相同的树”是异曲同工。

1.2.3 有序数组转换为二叉搜索树

将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。

思路：二叉搜索树的中序遍历为有序数组。可以将数组的中点作为树的根，数组的左边作为树的左子树，数组的右边即为根的右子树；那么操作顺序是，先生成根节点，再生成左右节点，本质为前序遍历。

    public TreeNode sortedArrayToBST(int[] nums){
      if(nums == null || nums.length == 0) return null;
      return build(nums, 0, nums.length-1);
    }
    private TreeNode build(int[] nums, int lo, int hi){
      if(lo > hi) return null;//临界条件
      
      int mid = (start + end) / 2;
      TreeNode root = new TreeNode(nums[mid]);//得到根节点
      root.left = build(nums, lo, mid-1);//得到左子树
      root.right = build(nums, mid+1, hi);//得到右子树
      
      return root;
    }



2.1 中序遍历

	中序遍历的节点访问顺序为，先访问左节点，再根节点，最后右节点；

递归法

    public List<Integer> inOrder(TreeNode root){
      List<Integer> ans = new ArrayList<>();
      if(root == null) return ans;
      dfs(root, ans);
      return ans;
    }
    public void dfs(TreeNode root, List<Integer> res){
      if(root == null) return; //递归终止条件
      dfs(root.left, res);
      res.add(root.val);
      dfs(root.right, res);
    }

非递归法

	中序遍历的非递归法，仍然需要一个栈来记录所访问的节点；和前序遍历的不同在于，中序遍历在探索完左子树后，出栈的节点为左子树最左端的的叶子节点，然后继续出栈，得到其父节点，再遍历其父节点的右节点；

    public List<Integer> inOrder(TreeNode root){
      List<Integer> ans = new ArrayList<>();
      if(root == null) return ans;
      Stack<TreeNode> s = new Stack<>();
      TreeNode cur = root;
      while(cur != null || !s.isEmpty()){
        if(cur != null){
          s.push(cur);
          cur = cur.left;//一直往左探索
        }else{
          TreeNode node = s.pop();
          ans.add(node.val);//处理根节点
          cur = node.right;//继续往右探索
        }
      }
      return ans;
    }

2.2 关于中序遍历的例子



3.1 后序遍历

	后序遍历的非递归是最难的。

递归法

    public List<Integer> postOrder(TreeNode root){
      List<Integer> ans = new ArrayList<>();
      if(root == null) return ans;
      dfs(root, ans)
      return ans;
    }
    public void dfs(TreeNode root, List<Integer> res){
      if(root == null) return ;
      dfs(root.left, res);
      dfs(root.right, res);
      res.add(root.val);
    }

非递归法

	后序遍历需要设置一个pre节点，用以控制输出根节点的时机，此外继续依赖栈，只是这里对栈的操作与前两种遍历对栈的使用大不一样；具体操作是，节点入栈顺序是先右节点后左节点，栈顶元素的取法是通过peek，而前序和中序遍历的方式是直接pop操作；

    public List<Integer> postOrder(TreeNode root){
      List<Integer> ans = new ArrayList<>();
      if(root == null) return ans;
      Stack<TreeNode> s = new Stack<>();
      s.push(root);
      TreeNode pre = null;
      while(!s.isEmpty()){
        TreeNode cur = s.peek();
        if(cur.left == null && cur.right == null //当前节点为叶子节点时，则直接输出cur
        ||(pre != null && (pre == cur.left && cur.right == null))//当前节点的左节点已经输出，其无右节点，直接输出cur
        ||(pre != null && pre == cur.right)//右节点已经输出，则输出根节点
        ){
          ans.add(cur.val);
          s.pop();
          pre = cur;
        }else{
          if(cur.right != null){
            s.push(cur.right);
          }
          if(cur.left != null){
            s.push(cur.left);
          }
        }
      }
      return ans;
    }

3.2关于后序遍历的例子



4.1 层次遍历

	层次遍历的顺序是一层一层的、从左往右地输出该层节点；这里的节点操作顺序，就有了前序遍历的影子了，即先输出第一层的根节点，再从左往右地输出下一层的节点；

递归法

    public List<List<Integer>> levelOrder(TreeNode root){
      List<List<Integer>> ans = new ArrayList<>();
      if(root == null) return ans;
      dfs(root, 0, ans);//第0层即为根节点
      return ans;
    }
    public void dfs(TreeNode root, int level, List<List<Integer>> ans){
      if(root == null) return;
      if(level == ans.size()){
        ans.add(new ArrayList<>);
      }
      ans.get(level).add(root.val);
      dfs(root.left, level+1, ans);
      dfs(root.right, level+1, ans);
    }

非递归法

	层序遍历的非递归法，用到了队列，将每层的节点按从左到右的顺序入队，此时队列的里所包含的元素即为该层的所有节点，逐一出队后，又对出队的节点的孩子节点继续入队，直到队列为空；

    public List<List<Integer>> levelOrder(TreeNode root){
      List<List<Integer>> ans = new ArrayList<>();
      if(root == null) return ans;
      Queue<TreeNode> q = new LinkedList<>();
      q.add(root);
      while(!q.isEmpty()){
        int size = q.size();
        List<Integer> res = new ArrayList<>();
        for(int i = 0; i < size; i ++){
          TreeNode node = q.poll();//出队
          res.add(node.val);//处理出队节点
          if(node.left != null) q.add(node.left); //处理左节点
          if(node.right != null) q.add(node.right);//处理右节点
        }
        ans.add(res);//将这一层的节点存入结果集中
      }
      return ans;
    }



1,二叉树的最小深度

和求二叉树的最大深度不一样的是，需要考虑两点：

1，当节点为叶子节点时，深度为1；

2，当二叉树只有单侧有节点时；

循环法

通过广度遍历，如果碰到叶子节点时，则停止遍历，返回深度；那么什么时候对变量depth赋值呢？判断是否遍历到树的最右端节点；当遍历到树的最右端节点时，说明这层已经遍历完全了，接下来就是遍历下一层了，此时depth++，并且重新赋值最右端节点；

    public int minDepth(TreeNode root){
      if(root == null) return 0;
      Queue<TreeNode> q = new LinkedList<>();
      q.add(root);
      TreeNode rightMost = root;
      int depth = 1;
      while(!q.isEmpty()){
         TreeNode node = q.poll();
        if(node.left == null && node.right == null) break;
        if(node.left != null) q.add(node.left);
        if(node.right != null) q.add(node.right);
        if(node == rightMost){
           depth ++;
          rightMost = (node.right != null) ? node.right : node.left;
        }
      }
      return depth;
    }

递归法

    public int minDepth(TreeNode root){
      if(root == null) return 0;
      if(root.left == null) return (minDepth(root.right)+1);
      if(root.right == null) return minDepth(root.left) + 1;
      return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
    }

2，平衡二叉树

平衡二叉树的定义为，每个节点的左右子树的高度差不超过1；

    public boolean isBalanced(TreeNode root){
        if(root == null) return true;
      //判断根节点的平衡因子是否大于1
      int LeftDepth = getDepth(root.left);
      int rightDepth = getDepth(root.right);
      if(Math.abs(leftDepth - rightDepth) > 1){
          return false;
      }else{
          //如果根节点平衡因子正常，则判断根节点的左右子树是否平衡
        return isBalanced(root.left) && isBalanced(root.right);
      }
    }
    private static int getDepth(TreeNode root){
        if(root == null) return 0;
      int leftDepth = getDepth(root.left);
      int rightDepth = getDepth(root.right);
      return leftDepth > rightDepth ? leftDepth+1 : rightDepth+1;
    }

3，将有序数组转换为二叉搜索树

给定一个按照升序排列的有序数组，将之转换为一棵高度平衡二叉搜索树。

如[-10, -3, 0, 5, 9]

因为树的中序遍历是有序的，因此一个有序数组能够还原为一颗树。具体做法是：

数组的中点为根节点，左右为子树，以此递归，最终得到一棵树。

    public TreeNode sortedArrayToBST(int[] nums){
        return build(nums, 0, nums.length-1);
    }
    private TreeNode build(int[] nums, int start, int end){
        if(start > end) return null;
      int mid = (start + end) / 2;
      TreeNode root = new TreeNode(nums[mid]);
      root.left = build(nums, start, mid-1);
      root.right = build(nums, mid+1, end);
      return root;
    }

4，将有序链表转换为二叉树

这回将有序数组转换为有序链表

依照有序数组找中点的思路，链表也能够通过快慢指针找到其中点；

1，快慢指针法

    public TreeNode sortedListToBST(ListNode head){
      return build(head, null);
    }
    private TreeNode build(ListNode head, ListNode tail){
        if(head == tail) return null;
      ListNode fast = head;
      ListNode slow = head;
      while(fast != tail && fast.next != tail){
          slow = slow.next;
        fast = fast.next.next;
      }
      TreeNode root = new TreeNode(slow.val);//slow指向的为链表中点
      root.left = build(head, slow);
      root.right = build(slow.next, tail);
      return root;
    }

2，Bottom-Up 建树法

通常使用的建树法是Top-Down方法。当使用Top-Down法不能使得问题变得简单时，可以考虑Bottom-Up法；

    private ListNode list;
    public TreeNode sortedListToBST(ListNode head){
      int n = 0;
      ListNode p = head;
      while(p != null){
          n++;
        p = p.next;
      }
      list = head;
      return build(0, n-1);
    }
    private TreeNode build(int start, int end){
        if(start > end) return null; //叶子节点
      int mid = (start + end) / 2;
      TreeNode leftChild = build(start, mid-1);
      TreeNode parent = new TreeNode(list.val);
      parent.left = leftChild;
      list = list.next;
      parent.right = build(mid+1, end);
      return parent;
    }

5,求根到叶子节点数字之和

给定一个二叉树，它的每个节点都存放一个0-9的数字，每条从根到叶子节点的路径都代表一个数字。例如路径1->2->3代表数字123；

我的思路：相当于求解一个二叉树的所有路径；那么可以通过一个String变量，将每次访问的节点值包括进去，到了叶子节点时，再将这条路径存入List<String>变量中，最后遍历这个List变量，将所有的值取出来求和，代码如下：

    public int sumNumbers(TreeNode root){
        if(root == null) return 0;
      List<String> res = new ArrayList<>();
      String tmp = new String();
      dfs(root, res, tmp);
      int ans = 0;
      for(int i = 0; i < res.size(); i ++){
          ans += Integer.parseInt(res.get(i));
      }
      return ans;
    }
    public static void dfs(TreeNode root, List<String> res, String tmp){
        if(root == null) return ;
      tmp += root.val;
      if(root.left == null && root.right == null){
          res.add(tmp);
      }
      dfs(root.left, res, tmp);
      dfs(root.right, res, tmp);
      tmp = tmp.substring(0, tmp.length()-1);//遍历到叶子节点后，再访问这个叶节点的父母节点的右子节点
    }

效率更高的思路为：直接对每次访问节点的数字进行数学计算转换为整数num = num*10 + root.val，num的初始值为0，当访问到叶子节点后，需要去除num值的最后一位，通过num/10就能实现，代码如下：

    public int sumNumber(TreeNode root){
        if(root == null) return 0;
      int ans = 0;
      int num = 0;
      dfs(root, ans, num);
      return ans;
    }
    public static void dfs(TreeNode root, int ans, int num){
        if(root == null) return;
      num = num * 10 + root.val;
      if(root.left == null && root.right == null){
          ans += num;
      }
      dfs(root.left, ans, num);
      dfs(root.right, ans, num);
      num /= 10;
    }

6，删除二叉搜索树中的节点

给定一个二叉搜索树的根节点和一个key值，删除二叉树中的key对应的节点。

思路：要删除的节点可能存在的三种情况：1，恰好为叶子节点；2，根节点，这个根节点只存在单侧的子节点；3，左右都有子节点的根节点。

    public TreeNode deleteNode(TreeNode root, int key){
        if(root == null) return null;
      if(root.val > key){//key在左子树，那么左子树会被修改，root.left需要重新赋值
          root.left = deleteNode(root.left, key);
      }else if(root.val < key){//在右边，同理
          root.right = deleteNode(root.right, key);
      }else{
          //找到key值的节点
        if(root.left == null || root.right == null){
            //只有单侧的情况以及叶子节点的情况
          //只有单侧的情况时，只需将该单侧的父节点作为新的根节点
          //为叶子节点时，root直接为null
          root = root.left == null ? root.right : root.left;
        }else{
            //根节点的左右都有子节点的情况
          TreeNode head = root.right;//往右边子树找可替代的根节点
          while(head.left != null) head = head.left;//根据二叉搜索树，左<根<右，的性质，可替代的根节点在右子树的左边
          root.val = head.val;//将找到的替代节点值覆盖原根节点的值
          root.right = deleteNode(root.right, head.val);//往右边寻找，将替代节点删除。
        }
      }
      return root;//完成操作
    }

7，恢复二叉搜索树

已知二叉搜索树中的两个节点被错误地交换，请在不改变其结构的情况下，恢复这棵树。

思路：二叉搜索树的中序遍历结果是有序的。

1，循环法

维护一个栈，按照中序遍历的方式，更新prev节点和curr节点，并判断是否prev节点值大于curr节点值，若大于，则找到了第一个错误节点，first=prev，second节点有可能是当前节点curr。当再找到一个prev节点值大于curr节点值时，则找到了第二个错误节点，second=prev。

    public void recoverTree(TreeNode root){
        if(root == null) return;
      TreeNode first = null;
      TreeNode second = null;
      TreeNode prev = null;
      TreeNode curr = root;
      Stack<TreeNode> s = new Stack<>();
      while(curr != null || !s.isEmpty()){
          if(curr != null){
              s.push(curr);
            curr = curr.left;
          }else{
              curr = s.pop();
            if(prev != null && prev.val > curr.val){
                if(first == null)
                  first = prev;
              second = curr;
            }
            prev = curr;
            curr = curr.right;
          }
      }
      if(first != null && second != null){
          int tmp = first.val;
        first.val = second.val;
        second.val = tmp;
      }
    }

2，递归法

利用中序遍历的递归方法，能够大大简化代码。

    private TreeNode first = null;
    private TreeNode second = null;
    public void recoverTree(TreeNode root){
      if(root == null) return ;
      inOrder(root);
      int tmp = first.val;
      first.val = second.val;
      second.val = tmp;
    }
    public void inOrder(TreeNode root){
      if(root == null) return;
      inOrder(root.left);
      if(prev != null && prev.val > root.val){
        if(first == null) first = prev;
        second = root;
      }
      prev = root;
      inOrder(root.right);
    }


