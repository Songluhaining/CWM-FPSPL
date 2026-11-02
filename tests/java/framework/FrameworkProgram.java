package frameworkfixture;

public class FrameworkProgram {
    public int value;
    public static int state;
    public FrameworkProgram() { value = 7; }
    public static int read(FrameworkProgram p) { return p.value; }
    public static void write(FrameworkProgram p, int n) { p.value = n; }
    public static int nested(FrameworkProgram p) { return read(p); }
    public static void unexpected() { throw new IllegalArgumentException(); }
    public static void callback() { state = 91; }
    public static class CustomNPE extends NullPointerException {
        public StackTraceElement[] getStackTrace() {
            callback();
            return new StackTraceElement[] {new StackTraceElement("frameworkfixture.FrameworkProgram", "callback", "FrameworkProgram.java", 11)};
        }
    }
}
