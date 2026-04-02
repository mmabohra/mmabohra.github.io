import { useEffect, useRef, useState } from "react";
import { ArrowUpRight } from "lucide-react";
import Hls from "hls.js";
import { motion } from "framer-motion";
import gsap from "gsap";

export default function Hero() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoUrl = "https://stream.mux.com/Aa02T7oM1wH5Mk5EEVDYhbZ1ChcdhRsS2m1NYyx4Ua1g.m3u8";

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (Hls.isSupported()) {
      const hls = new Hls();
      hls.loadSource(videoUrl);
      hls.attachMedia(video);
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = videoUrl;
    }

    // GSAP Entrance
    const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
    tl.to(".name-reveal", {
      opacity: 1,
      y: 0,
      duration: 1.2,
      delay: 0.5,
    })
      .to(".blur-in", {
        opacity: 1,
        filter: "blur(0px)",
        y: 0,
        duration: 1,
        stagger: 0.1,
      }, "-=0.8");
  }, []);

  return (
    <section className="blur-in relative h-screen w-full overflow-hidden flex items-center justify-center">
      {/* Background Video */}
      <div className="absolute inset-0 z-0">
        <video
          ref={videoRef}
          autoPlay
          muted
          loop
          playsInline
          className="absolute top-1/2 left-1/2 min-w-full min-h-full object-cover -translate-x-1/2 -translate-y-1/2"
        />
        <div className="absolute inset-0 bg-black/20" />
        <div className="absolute bottom-0 left-0 right-0 h-48 bg-gradient-to-t from-bg to-transparent" />
      </div>

      {/* Hero Content */}
      <div className="relative z-10 text-center px-4 max-w-4xl">
        <div className="blur-in translate-y-5 opacity-0 flex items-center gap-3 mb-4">
          <div className="relative w-2 h-2">
            <div className="absolute inset-0 rounded-full bg-green-500 animate-ping opacity-75" />
            <div className="relative w-2 h-2 rounded-full bg-green-500" />
          </div>
          <span className="text-xs text-muted uppercase tracking-[0.2em]">Open to work</span>
        </div>

        <h1 className="name-reveal translate-y-12 opacity-0 text-6xl md:text-8xl lg:text-9xl font-display italic leading-[0.9] tracking-tight text-text-primary mb-6">
          Mustafa Bohra
        </h1>

        <RoleLine />

        <p className="blur-in translate-y-5 opacity-0 text-sm md:text-base text-muted max-w-md mx-auto mb-12">
        </p>

        <div className="blur-in translate-y-5 opacity-0 flex flex-wrap justify-center gap-4">
          <button className="group relative rounded-full text-sm px-7 py-3.5 bg-text-primary text-bg transition-all hover:scale-105 hover:bg-bg hover:text-text-primary">
            <span className="relative z-10">Resume</span>
            <div className="absolute inset-0 rounded-full accent-gradient opacity-0 group-hover:opacity-100 -m-[1px] transition-opacity" />
            <div className="absolute inset-0 rounded-full bg-bg m-[1px] opacity-0 group-hover:opacity-100 transition-opacity" />
            <span className="absolute inset-0 flex items-center justify-center z-20 opacity-0 group-hover:opacity-100 transition-opacity">Resume</span>
          </button>

          <button className="group relative rounded-full text-sm px-7 py-3.5 border-2 border-stroke bg-bg text-text-primary transition-all hover:scale-105 hover:border-transparent">
            <span className="relative z-10 inline-flex items-center gap-1">Say hi <ArrowUpRight className="w-3 h-3" /></span>
            <div className="absolute inset-0 rounded-full accent-gradient opacity-0 group-hover:opacity-100 -m-[2px] transition-opacity" />
            <div className="absolute inset-0 rounded-full bg-bg m-[0px] opacity-0 group-hover:opacity-100 transition-opacity" />
            <span className="absolute inset-0 inline-flex items-center justify-center z-20 opacity-0 group-hover:opacity-100 transition-opacity gap-1">Say hi <ArrowUpRight className="w-3 h-3" /></span>
          </button>
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-12 left-1/2 -translate-x-1/2 flex flex-col items-center gap-4">
        <span className="text-[10px] text-muted uppercase tracking-[0.2em]">SCROLL</span>
        <div className="w-px h-10 bg-stroke relative overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-full bg-text-primary animate-scroll-down" />
        </div>
      </div>
    </section>
  );
}

const roles = ["Analyst", "Security Researcher", "Founder", "Scholar"];

function RoleLine() {
  const [roleIndex, setRoleIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setRoleIndex((prev) => (prev + 1) % roles.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="blur-in translate-y-5 opacity-0 text-lg md:text-xl text-muted mb-4">
      The{" "}
      <span
        key={roleIndex}
        className="font-display italic text-text-primary animate-role-fade-in inline-block"
      >
        {roles[roleIndex]}
      </span>{" "}
      lives in Dubai.
    </div>
  );
}
