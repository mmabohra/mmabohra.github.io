import { useState, useEffect } from "react";
import { ArrowUpRight } from "lucide-react";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 100);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex justify-center pt-4 md:pt-6 px-4">
      <div
        className={`inline-flex items-center rounded-full backdrop-blur-md border border-white/10 bg-surface px-2 py-2 transition-all duration-300 ${scrolled ? "shadow-md shadow-black/40" : ""
          }`}
      >
        {/* Logo */}
        <div className="group relative w-9 h-9 rounded-full flex items-center justify-center transition-transform hover:scale-110">
          <div className="absolute inset-0 rounded-full accent-gradient animate-gradient-shift group-hover:rotate-180 transition-transform duration-700" />
          <div className="absolute inset-[1px] rounded-full bg-bg flex items-center justify-center">
            <span className="font-display italic text-text-primary text-[13px] translate-y-[1px]">JA</span>
          </div>
        </div>

        <div className="hidden md:block w-px h-5 bg-stroke mx-2" />

        {/* Nav Links */}
        <div className="flex items-center gap-1">
          {["Home", "Work"].map((item) => (
            <button
              key={item}
              className={`text-xs sm:text-sm rounded-full px-3 sm:px-4 py-1.5 sm:py-2 transition-colors ${item === "Home"
                  ? "text-text-primary bg-stroke/50"
                  : "text-muted hover:text-text-primary hover:bg-stroke/50"
                }`}
            >
              {item}
            </button>
          ))}
        </div>

      </div>
    </nav>
  );
}
