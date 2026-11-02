import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import org.junit.runner.JUnitCore;
import org.junit.runner.Request;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;

/** One original JUnit4 method; completion is a durable record, never exit=0 alone. */
public final class Phase03TestRunner {
    public static void main(String[] a) throws Exception {
        if (a.length != 3) throw new IllegalArgumentException("class method result.tsv");
        Result r = new JUnitCore().run(Request.method(Class.forName(a[0]), a[1]));
        PrintWriter w = new PrintWriter(new OutputStreamWriter(new FileOutputStream(a[2]), StandardCharsets.UTF_8));
        for (Failure f : r.getFailures()) {
            Throwable t = f.getException();
            String message = String.valueOf(t.getMessage());
            w.println("FAILURE\t" + t.getClass().getName() + "\t" + Base64.getEncoder().encodeToString(message.getBytes(StandardCharsets.UTF_8)));
        }
        w.println("COMPLETE\t" + a[0] + "." + a[1] + "\t" + r.getRunCount() + "\t" + r.getFailureCount() + "\t" + r.getIgnoreCount());
        w.close();
        System.exit(r.wasSuccessful() ? 0 : 1);
    }
}
