package phase03fixture;

public class Phase03Program {
    int field;
    public static int chain(int x) {
        int y = x + 1;
        int z = y * 2;
        return z;
    }
    public static int branch(int x) {
        if (x > 0) return chain(x);
        return -x;
    }
    public static int loop(int n) {
        int sum = 0;
        for (int i = 0; i < n; i++) sum = sum + i;
        return sum;
    }
    public static int recursive(int n) {
        if (n == 0) return 0;
        return n + recursive(n - 1);
    }
    public int alias() {
        Phase03Program other = this;
        other.field = 7;
        return this.field;
    }
    public static int array() {
        int[] a = new int[2];
        a[0] = 1;
        a[1] = 2;
        return a[0];
    }
    public static int optimized(int x) {
        if (true) return x + 1;
        return x - 100; // Deliberately absent from original bytecode.
    }
}
