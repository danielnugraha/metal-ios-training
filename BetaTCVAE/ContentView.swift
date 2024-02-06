//
//  ContentView.swift
//  BetaTCVAE
//
//  Created by Daniel Nugraha on 16.06.23.
//

import SwiftUI
import MetalPerformanceShaders

struct ContentView: View {
    @ObservedObject var model = AEModel()
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundColor(.accentColor)
            
            Button(action: {
                Task {
                    await model.runAutoencoder()
                }
            }) { Text("Run autoencoder") }
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
