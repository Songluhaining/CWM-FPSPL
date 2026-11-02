package nullfieldfixture;

public class NullFieldProgram {
    public int value;
    public NullFieldProgram link;
    public NullFieldProgram() { value = 7; }
    public static int read(NullFieldProgram p) { return p.value; }
    public static void write(NullFieldProgram p, int x) { p.value = x; }
    public static int sameRead() {
        NullFieldProgram p = null;
        try { return 100 + p.value; }
        catch (NullPointerException e) { return 11; }
    }
    public static int sameWrite() {
        NullFieldProgram p = null;
        try { p.value = 0; return 0; }
        catch (NullPointerException e) { return 12; }
    }
    public static int callerRead() {
        try { return read(null); }
        catch (NullPointerException e) { return 13; }
    }
    public static int callerWrite() {
        try { write(null, Integer.MIN_VALUE); return 0; }
        catch (NullPointerException e) { return 14; }
    }
    public static int aliasedNull() {
        NullFieldProgram holder = new NullFieldProgram();
        holder.link = null;
        NullFieldProgram alias = holder.link;
        try { return read(alias); }
        catch (NullPointerException e) { return holder.value; }
    }
    public static int broadCatch() {
        try { return read(null); }
        catch (RuntimeException e) { return 15; }
    }
    public static int withFinally() {
        try { return read(null); }
        finally { int sentinel = 3; }
    }
    public static int divide(int x) { return 1 / x; }
}
