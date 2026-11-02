package exceptionfixture;

public class ExceptionProgram {
    public static int normalTryCatch(int x) {
        try { return x + 1; }
        catch (IllegalArgumentException ex) { return -9; }
    }
    public static int sameMethodThrow() {
        try { throw new IllegalArgumentException("fixture"); }
        catch (IllegalArgumentException ex) { return 7; }
    }
    public static void throwingCallee() {
        throw new IllegalArgumentException("fixture");
    }
    public static int callerCatch() {
        try { throwingCallee(); return 0; }
        catch (IllegalArgumentException ex) { return 9; }
    }
    public static int normalFinally() {
        int result = 0;
        try { result = 2; }
        finally { result += 3; }
        return result;
    }
    public static void exceptionalFinally() {
        try { throwingCallee(); }
        finally { int marker = 1; marker++; }
    }
    public static int ordinaryStore() {
        Object first = new Object();
        Object second = first;
        return first == second ? 1 : 0;
    }
    public static int implicitDivide(int denominator) {
        try { return 4 / denominator; }
        catch (ArithmeticException ex) { return -1; }
    }
}
