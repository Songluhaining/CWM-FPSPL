import com.sun.source.tree.*;
import com.sun.source.util.*;
import javax.tools.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Parse only: never analyze, generate, execute or modify product/feature sources. */
public class Phase03SourceExport {
    static String q(String s) {
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"")
            .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") + "\"";
    }
    public static void main(String[] args) throws Exception {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        try (StandardJavaFileManager fm = compiler.getStandardFileManager(diagnostics, null, StandardCharsets.UTF_8);
             PrintWriter out = new PrintWriter(args[0], "UTF-8")) {
            JavacTask task = (JavacTask) compiler.getTask(null, fm, diagnostics,
                Arrays.asList("-proc:none", "-encoding", "UTF-8"), null,
                fm.getJavaFileObjects(Arrays.copyOfRange(args, 1, args.length)));
            Iterable<? extends CompilationUnitTree> units = task.parse();
            SourcePositions positions = Trees.instance(task).getSourcePositions();
            for (CompilationUnitTree unit : units) {
                final String path = new File(unit.getSourceFile().toUri()).getCanonicalPath();
                new TreePathScanner<Void, Void>() {
                    String clazz = "";
                    void emit(Tree node, String kind, String name, String type, String params, Tree body) {
                        long start = positions.getStartPosition(unit, node), end = positions.getEndPosition(unit, node);
                        long bs = body == null ? -1 : positions.getStartPosition(unit, body);
                        long be = body == null ? -1 : positions.getEndPosition(unit, body);
                        StringJoiner names = new StringJoiner(",");
                        if (node instanceof MethodTree)
                            for (VariableTree parameter : ((MethodTree) node).getParameters()) names.add(parameter.getName().toString());
                        String classKind=node instanceof ClassTree ? node.getKind().name() : "";
                        String nesting="";
                        if(node instanceof ClassTree) nesting=getCurrentPath().getParentPath().getLeaf() instanceof CompilationUnitTree ? "TOP_LEVEL" :
                            ((ClassTree)node).getModifiers().getFlags().contains(javax.lang.model.element.Modifier.STATIC) ? "STATIC_MEMBER" : "NONSTATIC_MEMBER";
                        out.println("{\"path\":" + q(path) + ",\"class\":" + q(clazz) + ",\"kind\":" + q(kind)
                            + ",\"name\":" + q(name) + ",\"type\":" + q(type) + ",\"params\":" + q(params)
                            + ",\"parameter_names\":" + q(names.toString())
                            + ",\"class_kind\":" + q(classKind) + ",\"class_nesting\":" + q(nesting)
                            + ",\"start\":" + start + ",\"end\":" + end + ",\"body_start\":" + bs + ",\"body_end\":" + be
                            + ",\"line\":" + unit.getLineMap().getLineNumber(start)
                            + ",\"end_line\":" + unit.getLineMap().getLineNumber(end - 1) + "}");
                    }
                    @Override public Void visitClass(ClassTree node, Void ignored) {
                        String parent = clazz;
                        // Anonymous/local classes require a binary-name resolver; fail closed.
                        if (node.getSimpleName().length() == 0 || (!parent.isEmpty() && getCurrentPath().getParentPath().getLeaf().getKind() != Tree.Kind.CLASS)) return null;
                        clazz = parent.isEmpty() ? (unit.getPackageName() == null ? "" : unit.getPackageName() + ".") + node.getSimpleName() : parent + "$" + node.getSimpleName();
                        emit(node, "class", node.getSimpleName().toString(), "", "", null);
                        super.visitClass(node, ignored); clazz = parent; return null;
                    }
                    @Override public Void visitMethod(MethodTree node, Void ignored) {
                        StringJoiner types = new StringJoiner(",");
                        for (VariableTree p : node.getParameters()) types.add(p.getType().toString());
                        emit(node, "method", node.getName().toString(), node.getReturnType() == null ? "void" : node.getReturnType().toString(), types.toString(), node.getBody());
                        return super.visitMethod(node, ignored);
                    }
                    @Override public Void visitVariable(VariableTree node, Void ignored) {
                        if (getCurrentPath().getParentPath().getLeaf() instanceof ClassTree)
                            emit(node, "field", node.getName().toString(), node.getType().toString(), "", node.getInitializer());
                        return super.visitVariable(node, ignored);
                    }
                    @Override public Void visitTry(TryTree node, Void ignored) {
                        emit(node, "try", "", "", "", node.getBlock());
                        if (node.getFinallyBlock() != null) emit(node.getFinallyBlock(), "finally", "", "", "", node.getFinallyBlock());
                        return super.visitTry(node, ignored);
                    }
                }.scan(unit, null);
            }
            for (Diagnostic<?> d : diagnostics.getDiagnostics())
                if (d.getKind() == Diagnostic.Kind.ERROR) throw new IllegalArgumentException(d.toString());
        }
    }
}
