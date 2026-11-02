import java.io.*;
import java.util.*;
import javassist.*;
import javassist.bytecode.*;
import java.nio.file.*;
import java.security.MessageDigest;

/** Exact normal-flow successors from ORIGINAL class bytes; exceptions explicitly exported. */
public final class Phase03StaticExport {
    private static String hash(String path) throws Exception {
        byte[] bytes=MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(Paths.get(path)));
        StringBuilder s=new StringBuilder();for(byte b:bytes)s.append(String.format("%02x",b & 255));return s.toString();
    }
    public static void main(String[] a) throws Exception {
        PrintWriter out = new PrintWriter(new OutputStreamWriter(new FileOutputStream(a[0]), "UTF-8"));
        out.println("CONTROL_SCHEMA\tnormal_path_v1");
        for (int k=1; k<a.length; k++) {
            ClassFile cf = new ClassFile(new DataInputStream(new FileInputStream(a[k])));
            out.println("CLASS\t"+cf.getName()+"\t"+cf.getSuperclass());
            out.println("CLASSMETA\t"+cf.getName()+"\t"+cf.getSourceFile()+"\t"+cf.getAccessFlags()+"\t"+cf.getMajorVersion()+"\t"+hash(a[k])+"\t"+new File(a[k]).getCanonicalPath());
            for (Object item : cf.getMethods()) {
                MethodInfo m = (MethodInfo)item; CodeAttribute ca=m.getCodeAttribute();
                if(ca==null) continue;
                String prefix=cf.getName()+"\t"+m.getName()+"\t"+m.getDescriptor();
                out.println("METHODMETA\t"+prefix+"\t"+m.getAccessFlags());
                ExceptionTable et=ca.getExceptionTable();
                out.println("EXCEPTIONTABLE\t"+prefix+"\t"+ca.getCodeLength()+"\t"+et.size());
                for(int h=0;h<et.size();h++) out.println("HANDLER\t"+prefix+"\t"+et.startPc(h)+"\t"+et.endPc(h)+"\t"+et.handlerPc(h));
                for(int h=0;h<et.size();h++) {
                    int type=et.catchType(h);
                    out.println("HANDLERDETAIL\t"+prefix+"\t"+h+"\t"+et.startPc(h)+"\t"+et.endPc(h)+"\t"+et.handlerPc(h)+"\t"+(type==0?"*":m.getConstPool().getClassInfo(type)));
                }
                CodeIterator ci=ca.iterator();
                while(ci.hasNext()) {
                    int b=ci.next(), op=ci.byteAt(b), next=ci.hasNext()?ci.lookAhead():-1;
                    if(op>=178 && op<=181) {
                        int f=ci.u16bitAt(b+1);ConstPool pool=m.getConstPool();
                        out.println("FIELD\t"+prefix+"\t"+b+"\t"+pool.getFieldrefClassName(f)+"\t"+pool.getFieldrefName(f)+"\t"+pool.getFieldrefType(f));
                    }
                    if(op==187) out.println("NEW\t"+prefix+"\t"+b+"\t"+m.getConstPool().getClassInfo(ci.u16bitAt(b+1)));
                    if(op>=182 && op<=184) {
                        int f=ci.u16bitAt(b+1);ConstPool pool=m.getConstPool();
                        // Default constructors use ordinary Methodref constants.
                        if(pool.getTag(f)==10) out.println("CALL\t"+prefix+"\t"+b+"\t"+pool.getMethodrefClassName(f)+"\t"+pool.getMethodrefName(f)+"\t"+pool.getMethodrefType(f));
                    }
                    List<Integer> succ=new ArrayList<Integer>();
                    if((op>=153 && op<=166)||op==198||op==199) {succ.add(next);succ.add(b+ci.s16bitAt(b+1));}
                    else if(op==167) succ.add(b+ci.s16bitAt(b+1));
                    else if(op==200) succ.add(b+ci.s32bitAt(b+1));
                    else if(op==170||op==171) {
                        int p=(b+4)&~3;succ.add(b+ci.s32bitAt(p));
                        if(op==170) {int lo=ci.s32bitAt(p+4),hi=ci.s32bitAt(p+8); for(int i=0;i<=hi-lo;i++)succ.add(b+ci.s32bitAt(p+12+4*i));}
                        else {int n=ci.s32bitAt(p+4);for(int i=0;i<n;i++)succ.add(b+ci.s32bitAt(p+12+8*i));}
                    } else if((op>=172 && op<=177)||op==191) succ.add(-1);
                    else if(op==168||op==169||op==201) {out.println("UNSUPPORTED\t"+prefix+"\tlegacy_subroutine");succ.add(-1);}
                    else succ.add(next);
                    StringBuilder s=new StringBuilder();for(int v:succ){if(s.length()>0)s.append(',');s.append(v);}
                    out.println("I\t"+prefix+"\t"+m.getLineNumber(b)+"\t"+b+"\t"+op+"\t"+next+"\t"+s);
                }
            }
        }
        out.close();
    }
}
