"""
شبیه‌سازی مونت‌کارلو پارامتر τ - نسخه ساده
Monte Carlo Simulation of Vacuum Tension Parameter τ - Simplified Version
"""

import numpy as np
import matplotlib.pyplot as plt
import time

class SimpleTauMC:
    """شبیه‌سازی ساده شبکه τ"""
    
    def __init__(self, L=8, T=0.1, J=149.267):
        self.L = L
        self.T = T
        self.J = J
        
        # مقداردهی اولیه شبکه
        self.tau = 0.969818 + np.random.normal(0, 0.01, (L, L, L))
        self.tau = np.clip(self.tau, 0.01, 1.0)
        
        # تاریخچه
        self.energy_history = []
        self.tau_history = []
    
    def local_energy(self, i, j, k):
        """انرژی محلی"""
        tau_ijk = self.tau[i, j, k]
        
        # انرژی خودی
        E_self = self.J * (1/tau_ijk - 1)
        
        # انرژی برهم‌کنش (همسایه‌های نزدیک)
        E_int = 0
        neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        
        for dx, dy, dz in neighbors:
            ni = (i + dx) % self.L
            nj = (j + dy) % self.L
            nk = (k + dz) % self.L
            E_int += 0.05 * (tau_ijk - self.tau[ni, nj, nk])**2
        
        return E_self + E_int
    
    def total_energy(self):
        """انرژی کل"""
        E_total = 0
        for i in range(self.L):
            for j in range(self.L):
                for k in range(self.L):
                    E_total += self.local_energy(i, j, k)
        return E_total / (self.L**3)
    
    def metropolis_sweep(self):
        """یک پیمایش Metropolis"""
        for _ in range(self.L**3):
            # انتخاب نقطه تصادفی
            i, j, k = np.random.randint(0, self.L, 3)
            tau_old = self.tau[i, j, k]
            
            # پیشنهاد جدید
            tau_new = tau_old + np.random.normal(0, 0.02)
            tau_new = np.clip(tau_new, 0.01, 1.0)
            
            # انرژی قبلی
            E_old = self.local_energy(i, j, k)
            
            # تست جدید
            old_val = self.tau[i, j, k]
            self.tau[i, j, k] = tau_new
            E_new = self.local_energy(i, j, k)
            
            # تصمیم Metropolis
            delta_E = E_new - E_old
            if delta_E < 0:
                pass  # قبول
            else:
                if np.random.random() >= np.exp(-delta_E/self.T):
                    self.tau[i, j, k] = old_val  # رد
    
    def simulate(self, n_sweeps=200):
        """اجرای شبیه‌سازی"""
        print("شروع شبیه‌سازی مونت‌کارلو...")
        print(f"اندازه شبکه: {self.L}×{self.L}×{self.L}")
        print(f"دما: T = {self.T} MeV")
        print(f"ثابت انرژی: J = {self.J} MeV")
        
        start_time = time.time()
        
        # ذخیره اولیه
        self.energy_history.append(self.total_energy())
        self.tau_history.append(np.mean(self.tau))
        
        # پیمایش‌ها
        for sweep in range(n_sweeps):
            self.metropolis_sweep()
            
            if sweep % 20 == 0:
                self.energy_history.append(self.total_energy())
                self.tau_history.append(np.mean(self.tau))
                
                if sweep % 100 == 0:
                    print(f"پیمایش {sweep}/{n_sweeps}")
        
        elapsed = time.time() - start_time
        print(f"\nزمان اجرا: {elapsed:.1f} ثانیه")
        print(f"انرژی نهایی: {self.energy_history[-1]:.4f} MeV/گره")
        print(f"τ نهایی: {self.tau_history[-1]:.6f}")
    
    def plot_results(self):
        """رسم نتایج"""
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # 1. توزیع τ
        axes[0,0].hist(self.tau.flatten(), bins=30, alpha=0.7, color='blue')
        axes[0,0].set_xlabel('τ')
        axes[0,0].set_ylabel('تعداد')
        axes[0,0].set_title('توزیع τ')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. تکامل انرژی
        axes[0,1].plot(self.energy_history, 'r-')
        axes[0,1].set_xlabel('گام')
        axes[0,1].set_ylabel('انرژی (MeV/گره)')
        axes[0,1].set_title('تکامل انرژی')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. تکامل τ
        axes[0,2].plot(self.tau_history, 'g-')
        axes[0,2].set_xlabel('گام')
        axes[0,2].set_ylabel('میانگین τ')
        axes[0,2].set_title('تکامل τ')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4-6. برش‌های شبکه
        slices = [
            (self.tau[:, :, self.L//2], f'برش z={self.L//2}'),
            (self.tau[self.L//2, :, :], f'برش x={self.L//2}'),
            (self.tau[:, self.L//2, :], f'برش y={self.L//2}')
        ]
        
        for idx, (slice_data, title) in enumerate(slices):
            ax = axes[1, idx]
            im = ax.imshow(slice_data, cmap='viridis', vmin=0.95, vmax=1.0)
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('tau_mc_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # آمارها
        print("\n" + "="*50)
        print("آمارهای نهایی:")
        print(f"میانگین τ: {np.mean(self.tau):.6f}")
        print(f"انحراف معیار τ: {np.std(self.tau):.6f}")
        print(f"حداقل τ: {np.min(self.tau):.6f}")
        print(f"حداکثر τ: {np.max(self.tau):.6f}")
        
        # مناطق فشرده (τ < 0.97)
        compressed = self.tau[self.tau < 0.97]
        if len(compressed) > 0:
            print(f"\nمناطق فشرده (τ < 0.97):")
            print(f"  تعداد گره‌ها: {len(compressed)}")
            print(f"  درصد کل: {100*len(compressed)/self.tau.size:.1f}%")
            print(f"  میانگین τ مناطق فشرده: {np.mean(compressed):.6f}")
        
        return {
            'tau_mean': np.mean(self.tau),
            'tau_std': np.std(self.tau),
            'energy': self.energy_history[-1]
        }

# اجرای شبیه‌سازی
if __name__ == "__main__":
    print("="*50)
    print("شبیه‌سازی مونت‌کارلو τ - مدل FFM")
    print("="*50)
    
    # پارامترها
    L = 8  # اندازه شبکه (کوچکتر = سریعتر)
    T = 0.1  # دما
    n_sweeps = 200  # تعداد پیمایش‌ها
    
    # ایجاد و اجرای شبیه‌سازی
    mc = SimpleTauMC(L=L, T=T)
    mc.simulate(n_sweeps=n_sweeps)
    
    # نمایش نتایج
    stats = mc.plot_results()
    
    print("\n" + "="*50)
    print("شبیه‌سازی کامل شد!")
    print("="*50)